# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# create isaac environment
from omni.isaac.gym.vec_env import VecEnvBase

env = VecEnvBase(headless=True)

# create task and register task
from sawyer_task import SawyerTask

task = SawyerTask(name="Sawyer")
env.set_task(task, backend="torch")

# import stable baselines
from stable_baselines3 import PPO

# create agent from stable baselines
model = PPO(
    "MlpPolicy",
    env,
    n_steps=300,
    batch_size=300,
    n_epochs=20,
    learning_rate=0.001,
    gamma=0,
    device="cuda:0",
    ent_coef=0,
    vf_coef=0.5,
    max_grad_norm=1.0,
    verbose=1,
    tensorboard_log="./sawyer_tensorboard",
)
model.learn(total_timesteps=500000)
model.save("ppo_sawyer")

env.close()