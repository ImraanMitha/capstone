import numpy as np
import matplotlib.pyplot as plt
from agents import *
from utils import *
from manipulator_environment import *
import time

def train_loop(models_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # hyperparamers
    hypers = {"num_epochs": 500,
                    "batch_size": 1024,
                    "policy_lr": 1e-3,
                    "critic_lr": 1e-4,
                    "gamma": 0.7,
                    "tau": 0.005,
                    "action_noise": None,
                    "g_noise_std": 0.005*np.pi,
                    "replay_buffer_size": int(1e6),
                    "hidden_units": 512,
                    "num_steps": 1000
                    }

    # inits
    env = Planar_Environment()
    agent = DDPGagent(env, device, hypers)
    rewards = []
    avg_rewards = []

    start_time = time.time()
    print(f'Beginning training on {device}')
    print(f'\n{hypers["num_epochs"]} epochs, {hypers["num_steps"]} steps, {hypers["batch_size"]} batch size, lr_p={hypers["policy_lr"]}, lr_c={hypers["critic_lr"]}, gamma={hypers["gamma"]}, tau={hypers["tau"]}, {hypers["hidden_units"]} hidden units')

    for episode in range(hypers["num_epochs"]):
        epoch_start_time = time.time()
        state = env.reset()
        # env.viz_arm()
        episode_reward = 0

        action_history = []
        reward_history = []

        for step in range(hypers["num_steps"]):
            action, action_no_noise = agent.get_action(state, step)
            new_state, reward, done = env.step(action)
            # env.viz_arm()

            action_history.append(action_no_noise)
            reward_history.append(reward)

            agent.replay_buffer.push(state, action, reward, new_state, done)
            
            if len(agent.replay_buffer) > hypers["batch_size"]:
                agent.update(hypers["batch_size"])
                    
            state = new_state

            episode_reward += reward

            if done:
                print(f'Completion in episode {episode} by step {step}, with episode average reward: {round(episode_reward / hypers["num_steps"], 3)}')
                break


        rewards.append(episode_reward / hypers["num_steps"])
        avg_rewards.append(np.mean(rewards[-10:]))

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


    print(f'\nTotal time for {hypers["num_epochs"]} epochs: {round(time.time()-start_time, 3)}')

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
    fig.suptitle(f'{hypers["num_epochs"]} epochs, {hypers["num_steps"]} steps, {hypers["batch_size"]} batch size, lr_p={hypers["policy_lr"]}, lr_c={hypers["critic_lr"]}, gamma={hypers["gamma"]}, tau={hypers["tau"]}, {hypers["hidden_units"]} hidden units')
    
    if models_path is not None:
        agent.save(models_path, rewards, avg_rewards, fig)
    else:
        print("Cant save models, no path provided")

    plt.show()

if __name__ == "__main__":
    train_loop("models")