#!/bin/bash

# Schedule execution of many runs
# Run from rlydoe subfolder with: bash bash/schedule.sh

# Cartpole-v1
python trainer-sb3.py learner=ppo
python trainer-sb3.py learner=sac

# FetchPush-v1
python trainer-sb3.py environment=fetchpickplace \
    learner=sac learner.policy_type=MultiInputPolicy
        
python trainer-sb3.py environment=fetchpickplace \
    learner=sac learner.policy_type=MultiInputPolicy learner.replay_buffer_class=None
python trainer-sb3.py environment=fetchpickplace \ 
    learner=ppo learner.policy_type=MultiInputPolicy
