from manipulator_environment import Planar_Environment
import numpy as np
from collections import deque
import gym
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
from main import *

############## straight arm ###############
device = 'cuda' if torch.cuda.is_available() else 'cpu'
hypers = {"num_epochs": 1000,
                "batch_size": 100,
                "policy_lr": 1e-7,
                "critic_lr": 0.01,
                "gamma": 0.9,
                "tau": 0.005,
                "action_noise": "G",
                "g_noise_std": 0.02,
                "replay_buffer_size": int(1e6),
                "hidden_units":100,
                "num_steps": 100,
                "configuration": [('R', 10), ('R', 10)],
                }

env = Planar_Environment(configuration=hypers['configuration'])
num_actions = env.action_dim
num_states = env.state_dim
action_bound = env.action_bound

agent = DDPGagent(action_bound, num_actions, num_states, device, hypers)
agent.load("models/0059", device)

# for i in range(500):
#     ret = eval_run(i, agent, env, hypers, verbose=True)

eval_run(0, agent, env, hypers, goal=[3.44, 18.29], verbose=True, plot=True)
###########################################
