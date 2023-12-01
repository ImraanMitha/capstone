import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

'''
Class to define the replay buffer used during training. Implements a buffer of specified
max size, defines a push() function to add experiences, and a sample() function to
sample a random batch of experiences.
'''
class ReplayBuffer:
    def __init__(self, buf_size):
        self.buf_size = buf_size
        self.buffer = deque(maxlen=buf_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)    

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return np.array(state_batch), np.array(action_batch), np.array(reward_batch), np.array(next_state_batch), np.array(done_batch)
    
    def __len__(self):
        return len(self.buffer)

'''
Class to implement  Ornstein-Uhlenbeck noise, not used right now,
the sigma are too large and I have my doubts about the implementation,
eg in add_noise(), max_sigma-min_sigma is always 0 with the implementation defaults.
'''
class OUNoise(object):
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_dim
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def add_noise(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return action + ou_state

'''
Plots of the actor/critic losses as well as the episodes reward, throughout the training.
'''
def end_plots(agent, rewards, avg_rewards, hypers):

    fig, axs = plt.subplots(3, 1, figsize=(10, 9))
    axs[0].plot(agent.policy_loss_history, label="policy loss")
    axs[0].grid(True)
    axs[0].set_title("Policy loss")

    axs[1].plot(agent.critic_loss_history, label="critic loss")
    axs[1].grid(True)
    axs[1].set_title("critic loss")

    axs[2].plot(rewards, label = "episode avg reward")
    axs[2].plot(avg_rewards, label = '10 episode sliding average')
    axs[2].grid(True)
    axs[2].legend()
    axs[2].set_title("episode avg rewards")
    fig.suptitle(f'{hypers["num_episodes"]} epochs, {hypers["num_steps"]} steps, {hypers["batch_size"]} batch size, lr_p={hypers["policy_lr"]}, lr_c={hypers["critic_lr"]}, gamma={hypers["gamma"]}, tau={hypers["tau"]}, {hypers["action_noise"]} noise')
    plt.show()
    return fig

'''
Plots of action (no noise) and reward throughout epoch as well as
a running plot of the average return through training so far.
'''
def epoch_summary(episode, epoch_action_history, epoch_reward_history, rewards, action_bound):
    action_dim = epoch_action_history.shape[1]
    fig, axs = plt.subplots(action_dim, 1, figsize=(20,9))
    # plot actions
    for i in range(action_dim):
        axs[0].plot(epoch_action_history[:, i], label=f'action[{i}]')

    # add action bounds to plot
    axs[0].plot(action_bound*np.ones_like(epoch_action_history[:, 0]), color='red', linewidth = 0.5)
    axs[0].plot(-action_bound*np.ones_like(epoch_action_history[:, 0]), color='red', linewidth = 0.5)
    axs[0].legend()
    axs[0].set_title("no noise actions")

    # plot reward per step
    axs[1].plot(epoch_reward_history)
    axs[1].set_title("reward")


    # plot return
    axs[2].plot(rewards)
    axs[2].set_title("episode avg rewards")
    
    fig.suptitle(f'episode {episode}')
    for ax in axs:
        ax.grid(True)
    plt.show()  

'''
Deprecated, use plot_episode instead. Can still be useful to visualize
one step but since it instantiates its own fig & axs it can only do one step
and env.viz_arm is sufficient if you dont need action and reward plots.
'''
def step_viz(step, epoch_action_history, epoch_reward_history, action_bound, reward, state, env):
    # block to visualize action, arm pose and reward through the epoch
    plt.ion()

    fig, axs = plt.subplots(env.action_dim, 1, figsize=(20,9))
    for i in range(env.action_dim):
        axs[0].plot(epoch_action_history[:, i], label=f'action[{i}]')

    axs[0].plot(action_bound*np.ones_like(epoch_action_history[:, 0]), color='red', linewidth = 0.5)
    axs[0].plot(-action_bound*np.ones_like(epoch_action_history[:, 0]), color='red', linewidth = 0.5)
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title("no noise actions")

    axs[1].plot(epoch_reward_history)
    axs[1].grid(True)
    axs[1].set_title("reward")

    env.viz_arm(axs[2])
    fig.suptitle(f'step:{step}, step_reward: {round(reward, 3)}, goal pos: {[round(value, 3) for value in state[-2:]]}, finger pos: {[round(value, 3) for value in env.joint_end_points[-1]]}')
    
    inp = input()
   
    plt.close()
    plt.ioff()
    if inp.lower() == 'exit':
        return False
    return True

'''
Plots the actions, reward and environment throughout the episode.
'''
def plot_episode(fig, axs, episode, step, epoch_action_history, epoch_reward_history, reward, state, env):
    plt.ion()
    for ax in axs:
        ax.clear()

    for i in range(env.action_dim):
        axs[0].plot(epoch_action_history[:, i], label=f'action[{i}]')

    axs[0].plot(env.action_bound*np.ones_like(epoch_action_history[:, 0]), color='red', linewidth = 0.5)
    axs[0].plot(-env.action_bound*np.ones_like(epoch_action_history[:, 0]), color='red', linewidth = 0.5)
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title("no noise actions")

    axs[1].plot(epoch_reward_history)
    axs[1].grid(True)
    axs[1].set_title("reward")

    env.viz_arm(axs[2])
    fig.suptitle(f'Episode: {episode}, step:{step}, step_reward: {round(reward, 3)}, goal pos: {[round(value, 3) for value in state[-2:]]}, finger pos: {[round(value, 3) for value in env.joint_end_points[-1]]}')
    plt.pause(0.001)
    plt.ioff()

'''
Performs evaluation runs, env is stepped with no noise actions, returns normalized return.
'''
def eval_run(run, agent, env, hypers, goal=None, plot=False, verbose=False):
    if plot:
        pe_fig, pe_axs = plt.subplots(3, 1, figsize=(20,11))

    episode_return = 0 
    epoch_action_history = np.empty((0,env.action_dim))
    epoch_reward_history = np.empty((0,))

    state, _ = env.reset(goal)
    for step in range(hypers["num_steps"]):

        action, action_no_noise = agent.get_action(state, step)
        new_state, reward, done, _, _ = env.step(action_no_noise, step)

        state = new_state
        episode_return += reward
        
        epoch_action_history = np.append(epoch_action_history, np.array([action_no_noise]), axis=0)
        epoch_reward_history = np.append(epoch_reward_history, reward)


        if plot:
            plot_episode(pe_fig, pe_axs, "EVAL", step, epoch_action_history, epoch_reward_history, reward, state, env)
        
        if done:
            # if plot:
            #     input()
            break
    if plot:
        input()
    normalized_return = episode_return/hypers['num_steps']/env.working_radius
    if verbose:
        print(f"Eval run {run}: normalized return: {round(normalized_return, 3)}, \tgoal was {[round(value, 2) for value in state[-2:]]} {f'|Completion in {step} steps|' if done else ''}")

    return normalized_return
