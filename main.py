import numpy as np
import matplotlib.pyplot as plt
from agents import *
from utils import *
from manipuator_environment import *


env = Planar_Environment()

agent = DDPGagent(env)
batch_size = 128
rewards = []
avg_rewards = []

for episode in range(50):
    if episode % 10 == 0:
        print(f"Episode: {episode}")

    state = env.reset()
    episode_reward = 0
    
    for step in range(500):
        action = agent.get_action(state, add_noise=True) #looks like im (maybe) always getting 0 for one of the actions? at least for the first step in first episode. why?
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
            print(f"episode: {episode}, reward: {np.round(episode_reward, 2)}, average _reward: {np.mean(rewards[-10:])} \n")
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
