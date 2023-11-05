import numpy as np
import matplotlib.pyplot as plt
from agents import *
from utils import *
from manipuator_environment import *

# hyperparamers
num_epochs = 50
batch_size = 256
policy_lr=1e-6
critic_lr=1e-5
gamma=0.5
tau=1e-2
noise_std = 0.01*np.pi
replay_buffer_size=50000
hidden_units = 256
num_steps = 500 # number of steps per episode

# inits
env = Planar_Environment()
agent = DDPGagent(env, hidden_size=hidden_units, actor_learning_rate=policy_lr, critic_learning_rate=critic_lr, gamma=gamma, tau=tau, noise_std=noise_std, replay_buffer_size=replay_buffer_size)
rewards = []
avg_rewards = []


# TODO: I think in order for the batch norming to work I need to set the model to eval and only set it to train when I am in agent.update?
for episode in range(num_epochs):
    if episode % 10 == 0:
        print(f"Episode: {episode}")

    state = env.reset()
    episode_reward = 0

    action_history = []
    reward_history = []

    for step in range(num_steps):
        action, no_noise = agent.get_action(state, add_noise=True)
        new_state, reward, done = env.step(action)

        action_history.append(no_noise)
        reward_history.append(reward)

        agent.replay_buffer.push(state, action, reward, new_state, done)
        
        if len(agent.replay_buffer) > batch_size:
            agent.update(batch_size)
                
        state = new_state

        episode_reward += reward

        if done:
            print(f"episode: {episode}, reward: {np.round(episode_reward, 2)}, average_reward: {np.mean(rewards[-10:])} \n")
            break


    # block of code to visualize the value of the action without noise and the reward (- distance to goal) throughout the epoch, as well as the per episode average reward (average - distance to goal)
    rewards.append(episode_reward / num_steps)
    avg_rewards.append(np.mean(rewards[-10:]))

    action_history = np.array(action_history)
    reward_history = np.array(reward_history)
    fig, axs = plt.subplots(3, 1, figsize=(20,9))

    axs[0].plot(action_history[:, 0], label='action[0]')
    axs[0].plot(action_history[:, 1], label='action[1]')
    axs[0].plot(np.pi*np.ones_like(action_history[:, 0]), color='red', linewidth = 0.5)
    axs[0].plot(-np.pi*np.ones_like(action_history[:, 0]), color='red', linewidth = 0.5)
    
    axs[1].plot(reward_history)

    axs[2].plot(rewards)
    
    axs[0].legend()

    axs[0].set_title("no noise actions")
    axs[1].set_title("reward")
    axs[2].set_title("episode avg rewards")
    fig.canvas.manager.set_window_title(f'episode {episode}')
    plt.show()  


plt.plot(rewards, label="Episode avg reward")
plt.plot(avg_rewards, label="10 Episode sliding average")
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.show()
