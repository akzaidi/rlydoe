import os
import fire
import datetime
from collections import deque, namedtuple

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.distributions import Categorical
import torch.nn.functional as F

import gym
import wandb
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# constants

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data

Memory = namedtuple(
    "Memory", ["state", "action", "action_log_prob", "reward", "done", "value"]
)
AuxMemory = namedtuple("Memory", ["state", "target_value", "old_values"])


class ExperienceDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, ind):
        return tuple(map(lambda t: t[ind], self.data))


def create_shuffled_dataloader(data, batch_size):
    ds = ExperienceDataset(data)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


# helpers


def exists(val):
    return val is not None


def normalize(t, eps=1e-5):
    return (t - t.mean()) / (t.std() + eps)


def update_network_(loss, optimizer):
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()


def init_(m):
    if isinstance(m, nn.Linear):
        gain = torch.nn.init.calculate_gain("tanh")
        torch.nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


# networks


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, num_actions), nn.Softmax(dim=-1)
        )

        self.value_head = nn.Linear(hidden_dim, 1)
        self.apply(init_)

    def forward(self, x):
        hidden = self.net(x)
        return self.action_head(hidden), self.value_head(hidden)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(init_)

    def forward(self, x):
        return self.net(x)


# agent


def clipped_value_loss(values, rewards, old_values, clip):
    value_clipped = old_values + (values - old_values).clamp(-clip, clip)
    value_loss_1 = (value_clipped.flatten() - rewards) ** 2
    value_loss_2 = (values.flatten() - rewards) ** 2
    return torch.mean(torch.max(value_loss_1, value_loss_2))


class PPG:
    def __init__(
        self,
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        epochs,
        epochs_aux,
        minibatch_size,
        lr,
        betas,
        lam,
        gamma,
        beta_s,
        eps_clip,
        value_clip,
        writer,
    ):
        self.actor = Actor(state_dim, actor_hidden_dim, num_actions).to(device)
        self.critic = Critic(state_dim, critic_hidden_dim).to(device)
        self.opt_actor = Adam(self.actor.parameters(), lr=lr, betas=betas)
        self.opt_critic = Adam(self.critic.parameters(), lr=lr, betas=betas)

        self.minibatch_size = minibatch_size

        self.epochs = epochs
        self.epochs_aux = epochs_aux

        self.lam = lam
        self.gamma = gamma
        self.beta_s = beta_s

        self.eps_clip = eps_clip
        self.value_clip = value_clip

        self.writer = writer

    def save(self):
        torch.save(
            {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()},
            f"./ppg.pt",
        )

    def load(self):
        if not os.path.exists("./ppg.pt"):
            return

        data = torch.load(f"./ppg.pt")
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])

    def learn(self, memories, aux_memories, next_state):
        # retrieve and prepare data from memory for training
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        masks = []
        values = []

        for mem in memories:
            states.append(mem.state)
            actions.append(torch.tensor(mem.action))
            old_log_probs.append(mem.action_log_prob)
            rewards.append(mem.reward)
            masks.append(1 - float(mem.done))
            values.append(mem.value)

        # calculate generalized advantage estimate
        next_state = torch.from_numpy(next_state).to(device)
        next_value = self.critic(next_state).detach()
        values = values + [next_value]

        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lam * masks[i] * gae
            returns.insert(0, gae + values[i])

        # convert values to torch tensors
        to_torch_tensor = lambda t: torch.stack(t).to(device).detach()

        states = to_torch_tensor(states)
        actions = to_torch_tensor(actions)
        old_values = to_torch_tensor(values[:-1])
        old_log_probs = to_torch_tensor(old_log_probs)

        rewards = torch.tensor(returns).float().to(device)

        # store state and target values to auxiliary memory buffer for later training
        aux_memory = AuxMemory(states, rewards, old_values)
        aux_memories.append(aux_memory)

        # prepare dataloader for policy phase training
        dl = create_shuffled_dataloader(
            [states, actions, old_log_probs, rewards, old_values], self.minibatch_size
        )

        # policy phase training, similar to original PPO
        for _ in range(self.epochs):
            for states, actions, old_log_probs, rewards, old_values in dl:
                action_probs, _ = self.actor(states)
                values = self.critic(states)
                dist = Categorical(action_probs)
                action_log_probs = dist.log_prob(actions)
                entropy = dist.entropy()

                # calculate clipped surrogate objective, classic PPO loss
                ratios = (action_log_probs - old_log_probs).exp()
                advantages = normalize(rewards - old_values.detach())
                surr1 = ratios * advantages
                surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = -torch.min(surr1, surr2) - self.beta_s * entropy
                self.writer.add_scalar("losses/policy_loss", policy_loss.mean().item())

                update_network_(policy_loss, self.opt_actor)

                # calculate value loss and update value network separate from policy network
                value_loss = clipped_value_loss(
                    values, rewards, old_values, self.value_clip
                )
                self.writer.add_scalar("losses/value_loss", value_loss.item())
                # self.writer.add_scalar("losses/entropy", entropy.item())

                update_network_(value_loss, self.opt_critic)

    def learn_aux(self, aux_memories):
        # gather states and target values into one tensor
        states = []
        rewards = []
        old_values = []
        for state, reward, old_value in aux_memories:
            states.append(state)
            rewards.append(reward)
            old_values.append(old_value)

        states = torch.cat(states)
        rewards = torch.cat(rewards)
        old_values = torch.cat(old_values)

        # get old action predictions for minimizing kl divergence and clipping respectively
        old_action_probs, _ = self.actor(states)
        old_action_probs.detach_()

        # prepared dataloader for auxiliary phase training
        dl = create_shuffled_dataloader(
            [states, old_action_probs, rewards, old_values], self.minibatch_size
        )

        # the proposed auxiliary phase training
        # where the value is distilled into the policy network, while making sure the policy network does not change the action predictions (kl div loss)
        for epoch in range(self.epochs_aux):
            for states, old_action_probs, rewards, old_values in tqdm(
                dl, desc=f"auxiliary epoch {epoch}"
            ):
                action_probs, policy_values = self.actor(states)
                action_logprobs = action_probs.log()

                # policy network loss copmoses of both the kl div loss as well as the auxiliary loss
                aux_loss = clipped_value_loss(
                    policy_values, rewards, old_values, self.value_clip
                )
                loss_kl = F.kl_div(
                    action_logprobs, old_action_probs, reduction="batchmean"
                )
                policy_loss = aux_loss + loss_kl
                self.writer.add_scalar("losses/aux_loss", aux_loss.item())
                self.writer.add_scalar("losses/kl_loss", loss_kl.item())

                update_network_(policy_loss, self.opt_actor)

                # paper says it is important to train the value network extra during the auxiliary phase
                values = self.critic(states)
                value_loss = clipped_value_loss(
                    values, rewards, old_values, self.value_clip
                )

                update_network_(value_loss, self.opt_critic)


# main


def main(
    env_name: str = "LunarLander-v2",
    num_episodes: int = 50000,
    max_timesteps: int = 500,
    actor_hidden_dim: int = 32,
    critic_hidden_dim: int = 256,
    minibatch_size: int = 64,
    lr: float = 0.0005,
    betas: Tuple = (0.9, 0.999),
    lam: float = 0.95,
    gamma: float = 0.99,
    eps_clip: float = 0.2,
    value_clip: float = 0.4,
    beta_s: float = 0.01,
    update_timesteps: float = 5000,
    num_policy_updates_per_aux: float = 32,
    epochs: int = 1,
    epochs_aux: int = 6,
    seed: int = None,
    render: bool = False,
    render_every_eps: int = 250,
    save_every: int = 1000,
    load: bool = False,
    monitor: bool = False,
    wandb_save: bool = False,
):

    args = locals()
    env = gym.make(env_name)
    logger.info(f"Building environment for {env_name}")
    exp_name = f"{env_name}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

    if monitor:
        env = gym.wrappers.Monitor(env, f"runs/{exp_name}", force=True)

    if wandb_save:
        wandb.init(
            project="ppg-experiments",
            sync_tensorboard=True,
            name=exp_name,
            monitor_gym=True,
        )

    writer = SummaryWriter(f"runs/{exp_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in args.items()])),
    )

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    memories = deque([])
    aux_memories = deque([])

    agent = PPG(
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        epochs,
        epochs_aux,
        minibatch_size,
        lr,
        betas,
        lam,
        gamma,
        beta_s,
        eps_clip,
        value_clip,
        writer=writer,
    )

    if load:
        agent.load()

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    time = 0
    updated = False
    num_policy_updates = 0

    for eps in tqdm(range(num_episodes), desc="episodes"):
        render_eps = render and eps % render_every_eps == 0
        state = env.reset()
        for timestep in range(max_timesteps):
            time += 1

            if updated and render_eps:
                env.render()

            state = torch.from_numpy(state).to(device)
            action_probs, _ = agent.actor(state)
            value = agent.critic(state)

            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            action = action.item()

            next_state, reward, done, infos = env.step(action)
            for info in infos:
                if "episode" in info.keys():
                    print(f"time={time}, episode_reward={info['episode']['r']}")
                    writer.add_scalar(
                        "charts/episode_reward", info["episode"]["r"], time
                    )
                    break
            memory = Memory(state, action, action_log_prob, reward, done, value)
            memories.append(memory)

            state = next_state

            if time % update_timesteps == 0:
                agent.learn(memories, aux_memories, next_state)
                num_policy_updates += 1
                memories.clear()

                if num_policy_updates % num_policy_updates_per_aux == 0:
                    agent.learn_aux(aux_memories)
                    aux_memories.clear()

                updated = True

            if done:
                if render_eps:
                    updated = False
                break

        if render_eps:
            env.close()

        if eps % save_every == 0:
            agent.save()

    writer.close()


if __name__ == "__main__":
    fire.Fire(main)
