#!/bin/bash

# Schedule execution of many runs
# Run from rlydoe subfolder with: bash bash/schedule.sh

let NUM_TIMESTEPS=4*10**7

# Cartpole-v1
python trainer-sb3.py learner=ppo learner.total_timesteps=$NUM_TIMESTEPS
# sac sb3 does not support discrete actions
# python trainer-sb3.py learner=sac

# FetchPush-v1
python trainer-sb3.py environment=fetchpickplace \
    learner=sac learner.policy_type=MultiInputPolicy learner.total_timesteps=$NUM_TIMESTEPS
        
python trainer-sb3.py environment=fetchpickplace \
    learner=sac learner.policy_type=MultiInputPolicy learner.replay_buffer_class=None learner.total_timesteps=$NUM_TIMESTEPS

python trainer-sb3.py environment=fetchpickplace \
    learner=ppo learner.policy_type=MultiInputPolicy learner.total_timesteps=$NUM_TIMESTEPS
