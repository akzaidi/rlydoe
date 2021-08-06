# for VMs or colab:
# Set up fake display; otherwise rendering will fail

import os
from typing import Callable

if not "DISPLAY" in os.environ:
    os.system("Xvfb :1 -screen 0 1024x768x24 &")
    os.environ["DISPLAY"] = ":1"

import datetime
import gym
import highway_env
import logging
from functools import partial
from stable_baselines3 import PPO
from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecVideoRecorder,
    SubprocVecEnv,
)
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
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
    # and monitor stats
    # def make_env(env_name: str, num_instances: int = 1):

    #     env = gym.make(env_name)
    #     env = Monitor(env)

    #     return env
    def make_env(
        env_name: str, rank: int = 0, seed: int = 0, multi_proc: bool = False
    ) -> Callable:
        """
        Utility function for multiprocessed env.
        
        :param env_name: (str) the environment ID
        :param num_env: (int) the number of environment you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        :return: (Callable)
        """

        if multi_proc:

            def _init() -> gym.Env:
                env = gym.make(env_name)
                env = Monitor(env)
                env.seed(seed + rank)
                return env

            set_random_seed(seed)
            return _init
        else:
            env = gym.make(env_name)
            env = Monitor(env)
            return env

    # vectorize environment and record videos periodically
    num_instances = int(cfg["environment"]["instances"])
    if num_instances > 1:
        logger.info(f"Asked to run {num_instances} using async")
        # env = SubprocVecEnv(
        #     [
        #         make_env(env_name=cfg["environment"]["name"], rank=i, multi_proc=True)
        #         for i in range(num_instances)
        #     ]
        # )
        env = make_vec_env(cfg["environment"]["name"], n_envs=num_instances)
    else:
        pmake_env = partial(make_env, env_name=cfg["environment"]["name"])
        env = DummyVecEnv([pmake_env])
    env = VecVideoRecorder(
        env, "videos", record_video_trigger=lambda x: x % 2000 == 0, video_length=200
    )

    # define the algorithm, automatically upper case for more prominent display
    algorithm = cfg["learner"]["name"].upper()

    # experiment name for tracking
    if cfg["callbacks"]["experiment_name"] == "default":
        experiment_name = f"{cfg['environment']['name']}_{datetime.datetime.now().strftime('%Y-%m-%m-%H:%M:%S')}_{algorithm}"
    else:
        experiment_name = cfg["callbacks"]["experiment_name"]

    # define list of callbacks here
    callback_list = []

    if cfg["callbacks"]["wandb"]:
        wandb.init(
            name=experiment_name,
            project="sb3",
            config=cfg,
            sync_tensorboard=cfg["callbacks"][
                "sync_tensorboard"
            ],  # auto-upload sb3's tensorboard metrics
            monitor_gym=cfg["callbacks"][
                "monitor_gym"
            ],  # auto-upload the videos of agents playing the game
            save_code=cfg["callbacks"]["save_code"],  # optional
        )
        callback_list.append(
            WandbCallback(
                gradient_save_freq=100,
                model_save_freq=1000,
                model_save_path=f"models/{experiment_name}",
            )
        )

    if cfg["callbacks"]["early_stopping"]:
        if cfg["environment"]["max_reward"] is None:
            logger.error(
                f"Max reward must be provied in conf.environment.max_reward for early stopping"
            )
        else:
            logger.info(
                f"Early stopping when reward reaches {cfg['environment']['max_reward']}"
            )
            # eval_env = gym.make(cfg["environment"]["name"])
            peval_env = partial(make_env, env_name=cfg["environment"]["name"])
            eval_env = DummyVecEnv([peval_env])
            eval_env = VecVideoRecorder(
                eval_env,
                "videos",
                record_video_trigger=lambda x: x % 2000 == 0,
                video_length=200,
                name_prefix="eval",
            )
            n_eval_episodes = 100
            callback_on_best = StopTrainingOnRewardThreshold(
                reward_threshold=cfg["environment"]["max_reward"] * n_eval_episodes,
                verbose=1,
            )
            eval_callback = EvalCallback(
                eval_env,
                n_eval_episodes=n_eval_episodes,
                callback_on_new_best=callback_on_best,
                verbose=1,
            )
            callback_list.append(eval_callback)

    # instantiate model class
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
            replay_buffer_kwargs = cfg["learner"]["replay_buffer_kwargs"]
        else:
            replay_buffer_class = None
            replay_buffer_kwargs = None

        model = SAC(
            cfg["learner"]["policy_type"],
            env,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            verbose=1,
            tensorboard_log=f"runs/{experiment_name}",
        )
    else:
        raise ValueError(f"Unknown learner: {cfg['learner']['name']}")

    model.learn(
        total_timesteps=cfg["learner"]["total_timesteps"]
        * cfg["environment"]["instances"],
        callback=callback_list,
    )
    model.save(f"models/{experiment_name}")


if __name__ == "__main__":

    run_trainer()
