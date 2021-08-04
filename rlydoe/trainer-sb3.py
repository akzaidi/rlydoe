# for VMs or colab:
# Set up fake display; otherwise rendering will fail

import os

if not "DISPLAY" in os.environ:
    os.system("Xvfb :1 -screen 0 1024x768x24 &")
    os.environ["DISPLAY"] = ":1"

import datetime
import gym
import logging
from functools import partial
from stable_baselines3 import PPO
from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback
import wandb

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("sb3-gym")


@hydra.main(config_path="conf", config_name="config")
def run_trainer(cfg: DictConfig):

    logger.info("Configuration: ")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # define the environment
    def make_env(env_name):

        env = gym.make(env_name)
        env = Monitor(env)  # record stats such as returns
        return env

    # define the algorithm, automatically upper case for richer display
    algorithm = cfg["learner"]["name"].upper()

    if cfg["callbacks"]["experiment_name"] == "default":
        experiment_name = f"{cfg['environment']['name']}_{datetime.datetime.now().strftime('%Y-%m-%m-%H:%M:%S')}_{algorithm}"
    else:
        experiment_name = cfg["callbacks"]["experiment_name"]

    callback_list = []

    if cfg["callbacks"]["wandb"]:
        wandb.init(
            name=experiment_name,
            project="sb3",
            config=cfg,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        callback_list.append(
            WandbCallback(
                gradient_save_freq=100,
                model_save_freq=1000,
                model_save_path=f"models/{experiment_name}",
            )
        )

    pmake_env = partial(make_env, env_name=cfg["environment"]["name"])
    env = DummyVecEnv([pmake_env])
    env = VecVideoRecorder(
        env, "videos", record_video_trigger=lambda x: x % 2000 == 0, video_length=200
    )  # record videos

    if cfg["learner"]["name"].lower() == "ppo":
        model = PPO(
            cfg["learner"]["policy_type"],
            env,
            verbose=1,
            tensorboard_log=f"runs/{experiment_name}",
        )
    elif cfg["learner"]["name"].lower() == "sac":
        if cfg["learner"]["replay_buffer_class"].lower() == "her":
            replay_buffer_class = HerReplayBuffer
        else:
            replay_buffer_class = None

        model = SAC(
            cfg["learner"]["policy_type"],
            env,
            replay_buffer_class=replay_buffer_class,
            verbose=1,
            tensorboard_log=f"runs/{experiment_name}",
        )
    else:
        raise ValueError(f"Unknown learner: {cfg['learner']['name']}")

    model.learn(
        total_timesteps=cfg["learner"]["total_timesteps"], callback=callback_list,
    )
    model.save(f"models/{experiment_name}")


if __name__ == "__main__":

    run_trainer()
