import numpy as np
import matplotlib.pyplot as plt
from agents import *
from utils import *
from manipulator_environment import *
import time

def train_loop():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # hyperparamers
    num_epochs = 200
    batch_size = 1024
    policy_lr=5e-2
    critic_lr=1e-4
    gamma=0.9
    tau=0.005
    noise_std = 0.005*np.pi
    replay_buffer_size=int(1e6)
    hidden_units = 512
    num_steps = 1000 # number of steps per episode

    # inits
    env = Planar_Environment()
    agent = DDPGagent(env, device, hidden_size=hidden_units, actor_learning_rate=policy_lr, critic_learning_rate=critic_lr, gamma=gamma, tau=tau, noise_std=noise_std, replay_buffer_size=replay_buffer_size)
    rewards = []
    avg_rewards = []

    start_time = time.time()
    print(f'Beginning training on {device}')
    print(f'\n{num_epochs} epochs, {num_steps} steps, {batch_size} batch size, lr_p={policy_lr}, lr_c=c{critic_lr}, gamma={gamma}, tau={tau}, {hidden_units} hidden')

    for episode in range(num_epochs):
        epoch_start_time = time.time()
        state = env.reset()
        # env.viz_arm()
        episode_reward = 0

        action_history = []
        reward_history = []

        # agent.policy_loss_history = []
        # agent.critic_loss_history = []

        for step in range(num_steps):
            action, action_no_noise = agent.get_action(state, step, add_noise="G")
            new_state, reward, done = env.step(action)
            # env.viz_arm()

            action_history.append(action_no_noise)
            reward_history.append(reward)

            agent.replay_buffer.push(state, action, reward, new_state, done)
            
            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)
                    
            state = new_state

            episode_reward += reward

            if done:
                print(f'Completed in episode {episode} by step {step}, with episode average reward: {round(episode_reward / num_steps, 3)}')
                break


        rewards.append(episode_reward / num_steps)
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


    print(f'\nTotal time for {num_epochs} epochs: {round(time.time()-start_time, 3)}')
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
    fig.canvas.manager.set_window_title(f'{num_epochs} epochs, {num_steps} steps, {batch_size} batch size, lr_p={policy_lr}, lr_c=c{critic_lr}, gamma={gamma}, tau={tau}, {hidden_units} hidden')

    print(f'\n{num_epochs} epochs, {num_steps} steps, {batch_size} batch size, lr_p={policy_lr}, lr_c=c{critic_lr}, gamma={gamma}, tau={tau}, {hidden_units} hidden')

    plt.show()


    # standard plotting of epoch avg rewards and 10 epoch sliding average
    # plt.plot(rewards, label="Episode avg reward")
    # plt.plot(avg_rewards, label="10 Episode sliding average")
    # plt.legend()
    # plt.xlabel('Episode')
    # plt.ylabel('Average Reward')
    # plt.show()

if __name__ == "__main__":
    train_loop()