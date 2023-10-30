import numpy as np
import matplotlib.pyplot as plt
from agents import *
from utils import *
from manipuator_environment import *

# hyperparamers
num_epochs = 50
batch_size = 128
policy_lr=1e-6
critic_lr=1e-6

# inits
env = Planar_Environment()
agent = DDPGagent(env, actor_learning_rate=policy_lr, critic_learning_rate=critic_lr)
rewards = []
avg_rewards = []

for episode in range(num_epochs):
    if episode % 10 == 0:
        print(f"Episode: {episode}")

    state = env.reset()
    episode_reward = 0
    
    for step in range(500):
        action = agent.get_action(state, add_noise=True)
       
        ## periodiclly prints action we get to observe the values
        # if (step+1) % 100 == 0:
        #     print(f"step {step}: {action}", end=" ")
        #     if step == 499:
        #         print("\n")

        new_state, reward, done = env.step(action)
        agent.replay_buffer.push(state, action, reward, new_state, done)
        
        if len(agent.replay_buffer) > batch_size:
            agent.update(batch_size)
                
        state = new_state

        episode_reward += reward

        if done:
            print(f"episode: {episode}, reward: {np.round(episode_reward, 2)}, average_reward: {np.mean(rewards[-10:])} \n")
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

plt.plot(rewards, label="Episode cum. reward")
plt.plot(avg_rewards, label="10 Episode sliding average")
# plt.plot()
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
