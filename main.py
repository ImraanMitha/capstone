import numpy as np
import matplotlib.pyplot as plt
from agents import *
from utils import *
from manipulator_environment import *
import time
import gym

def train_loop(hypers, models_path=None, save=True):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # environment setup
    env = Planar_Environment()
    num_actions = env.action_dim
    num_states = env.state_dim
    action_bound = env.action_bound

    # env = gym.make("Pendulum-v1")
    # num_actions = env.action_space.shape[0]
    # num_states = env.observation_space.shape[0]
    # action_bound = env.action_space.high


    agent = DDPGagent(action_bound, num_actions, num_states, device, hypers)
    rewards = []
    avg_rewards = []

    start_time = time.time()
    print(f'Beginning training on {device}')
    print(f'\n{hypers["num_epochs"]} epochs, {hypers["num_steps"]} steps, {hypers["batch_size"]} batch size, lr_p={hypers["policy_lr"]}, lr_c={hypers["critic_lr"]}, gamma={hypers["gamma"]}, tau={hypers["tau"]}, {hypers["action_noise"]} noise')

    for episode in range(hypers["num_epochs"]):
        epoch_start_time = time.time()
        state, _ = env.reset()
        episode_reward = 0

        action_history = []
        reward_history = []

        for step in range(hypers["num_steps"]):
            action, action_no_noise = agent.get_action(state, step)
            new_state, reward, done, _, _ = env.step(action)

            action_history.append(action_no_noise)
            reward_history.append(reward)

            agent.replay_buffer.push(state, action, reward, new_state, done)
            
            # if len(agent.replay_buffer) > hypers["batch_size"]:
            #     agent.update(hypers["batch_size"])
                    
            state = new_state

            episode_reward += reward

            if done:
                print(f'Completion in episode {episode} by step {step}, with episode average reward: {round(episode_reward / hypers["num_steps"], 3)}')
                break


        rewards.append(episode_reward / hypers["num_steps"])
        avg_rewards.append(np.mean(rewards[-10:]))
        
        for i in range(200):
            agent.update(hypers["batch_size"])

        print(f'Episode {episode}, average reward: {round(rewards[-1], 3)}, in {round(time.time()-epoch_start_time, 3)} s')

        # block of code to visualize the value of the action without noise and the reward (- distance to goal) throughout the epoch, as well as the per episode average reward (average -distance to goal)
        # action_history = np.array(action_history)
        # reward_history = np.array(reward_history)
        # fig, axs = plt.subplots(3, 1, figsize=(20,9))
        # axs[0].plot(action_history[:, 0], label='action[0]')
        # axs[0].plot(action_history[:, 1], label='action[1]')
        # axs[0].plot(0.5*np.ones_like(action_history[:, 0]), color='red', linewidth = 0.5)
        # axs[0].plot(-0.5*np.ones_like(action_history[:, 0]), color='red', linewidth = 0.5)
        # axs[1].plot(reward_history)
        # axs[2].plot(rewards)
        # axs[0].legend()
        # axs[0].set_title("no noise actions")
        # axs[1].set_title("reward")
        # axs[2].set_title("episode avg rewards")
        # fig.canvas.manager.set_window_title(f'episode {episode}')
        # for ax in axs:
        #     ax.grid(True)
        # plt.show()  

    total_time = time.time()-start_time
    (hours, minutes), seconds  = divmod(divmod(total_time, 60)[0], 60), (divmod(total_time, 60)[1])
    print(f'\nTotal time for {hypers["num_epochs"]} epochs: {int(hours)}h {int(minutes)}m {round(seconds, 2)}s')

    # code to visualize the actor/critic losses throughout the episode as well as the episodes reward
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
    fig.suptitle(f'{hypers["num_epochs"]} epochs, {hypers["num_steps"]} steps, {hypers["batch_size"]} batch size, lr_p={hypers["policy_lr"]}, lr_c={hypers["critic_lr"]}, gamma={hypers["gamma"]}, tau={hypers["tau"]}, {hypers["action_noise"]} noise')

    if save:    
        if models_path is not None:
            agent.save(models_path, rewards, avg_rewards, fig)
        else:
            print("Cant save models, no path provided")

    plt.show()

if __name__ == "__main__":

    hypers = {"num_epochs": 10,
                    "batch_size": 100,
                    "policy_lr": 0.01,
                    "critic_lr": 0.001,
                    "gamma": 0.99,
                    "tau": 0.005,
                    "action_noise": None,
                    "g_noise_std": 0.1,
                    "replay_buffer_size": int(1e6),
                    "hidden_units": 512, # not currently used
                    "num_steps": 1000
                    }
    
    train_loop(hypers, models_path="models", save=False)