from manipulator_environment import Planar_Environment
import numpy as np
from collections import deque
import gym
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
from main import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'
hypers = {"num_epochs": 1000,
                "batch_size": 100,
                "policy_lr": 1e-6,
                "critic_lr": 0.01,
                "gamma": 0.9,
                "tau": 0.005,
                "action_noise": "G",
                "g_noise_std": 0.02,
                "replay_buffer_size": int(1e6),
                "hidden_units":100,
                "num_steps": 100,
                "configuration": [('R', 10), ('R', 10), ('R', 10)],
                }

env = Planar_Environment(configuration=hypers['configuration'])
num_actions = env.action_dim
num_states = env.state_dim
action_bound = env.action_bound

agent = DDPGagent(action_bound, num_actions, num_states, device, hypers)
agent.load("models/0078", device)

# ret = eval_run(agent, env, hypers, plot=True)
# print(ret)





# num_evals = 1000
# eval_reward = 0
# worst_ret = 0
# eval_start_time = time.time()
# for run in range(num_evals):
#     ret = eval_run(agent, env, hypers, verbose=False, plot=False)
#     eval_reward = eval_reward + ret
#     if ret < worst_ret:
#         worst_ret = ret

# print(f'worst episode return: {worst_ret}')

# eval_performance = eval_reward / num_evals # average avg dist to goal across evaluation runs
# print(f'Average reward over {num_evals} eval runs: {round(eval_performance, 4)} in {round(time.time()-eval_start_time, 2)} s')


for i in range(10):
    ret = eval_run(agent, env, hypers, plot=True)
    print(ret)
    # plt.show()
