import numpy as np
import matplotlib.pyplot as plt
from agents import *
from utils import *
from manipulator_environment import *
import time
import gym

def end_plots(agent, rewards, avg_rewards, hypers):
    '''
    Plots of the actor/critic losses as well as the episodes reward, throughout the training
    '''
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
    plt.show()
    return fig

def epoch_summary(episode, epoch_action_history, epoch_reward_history, rewards, action_bound):
    '''
    Plots of action (no noise) and reward throughout epoch as well as a running plot of the average epoch reward through training so far
    '''
    fig, axs = plt.subplots(3, 1, figsize=(20,9))
    axs[0].plot(epoch_action_history[:, 0], label='action[0]')
    axs[0].plot(epoch_action_history[:, 1], label='action[1]')
    axs[0].plot(action_bound*np.ones_like(epoch_action_history[:, 0]), color='red', linewidth = 0.5)
    axs[0].plot(-action_bound*np.ones_like(epoch_action_history[:, 0]), color='red', linewidth = 0.5)
    axs[1].plot(epoch_reward_history)
    axs[2].plot(rewards)
    axs[0].legend()
    axs[0].set_title("no noise actions")
    axs[1].set_title("reward")
    axs[2].set_title("episode avg rewards")
    fig.suptitle(f'episode {episode}')
    for ax in axs:
        ax.grid(True)
    plt.show()  

def step_viz(step, epoch_action_history, epoch_reward_history, action_bound, reward, state, env):
    # block to visualize action, arm pose and reward through the epoch
    plt.ion()
    fig, axs = plt.subplots(3, 1, figsize=(20,9))
    axs[0].plot(epoch_action_history[:, 0], label='action[0]')
    axs[0].plot(epoch_action_history[:, 1], label='action[1]')
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

def plot_epoch(fig, axs, episode, step, epoch_action_history, epoch_reward_history, reward, state, env):
    plt.ion()
    for ax in axs:
        ax.clear()

    axs[0].plot(epoch_action_history[:, 0], label='action[0]')
    axs[0].plot(epoch_action_history[:, 1], label='action[1]')
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

def eval_run(agent, env, hypers, plot=False):
    if plot:
        pe_fig, pe_axs = plt.subplots(3, 1, figsize=(20,9))

    episode_reward = 0 
    epoch_action_history = np.empty((0,2))
    epoch_reward_history = np.empty((0,))

    state, _ = env.reset()
    for step in range(hypers["num_steps"]):
        action, action_no_noise = agent.get_action(state, step)
        new_state, reward, done, _, _ = env.step(action_no_noise, step)
                
        state = new_state
        episode_reward += reward
        
        epoch_action_history = np.append(epoch_action_history, np.array([action_no_noise]), axis=0)
        epoch_reward_history = np.append(epoch_reward_history, reward)

        if done:
            break

        if plot:
            plot_epoch(pe_fig, pe_axs, "EVAL", step, epoch_action_history, epoch_reward_history, reward, state, env)

    print(f"Completed: {done}, took {step} steps, average reward: {round(episode_reward/hypers['num_steps'], 3)}, goal was {[round(value, 2) for value in state[-2:]]}")
    return episode_reward/hypers['num_steps']

def train_loop(hypers, models_path=None, save=True):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # environment setup
    env = Planar_Environment(configuration=hypers['configuration'])
    num_actions = env.action_dim
    num_states = env.state_dim
    action_bound = env.action_bound

    # env = gym.make("Pendulum-v1")
    # num_actions = env.action_space.shape[0]
    # num_states = env.observation_space.shape[0]
    # action_bound = env.action_space.high

    # env = gym.make("Reacher-v4")
    # num_actions = env.action_space.shape[0]
    # num_states = env.observation_space.shape[0]
    # action_bound = env.action_space.high

    agent = DDPGagent(action_bound, num_actions, num_states, device, hypers)
    rewards = []
    avg_rewards = []

    start_time = time.time()
    print(f'Beginning training on {device}')
    print(f'\n{hypers["num_epochs"]} epochs, {hypers["num_steps"]} steps, {hypers["batch_size"]} batch size, lr_p={hypers["policy_lr"]}, lr_c={hypers["critic_lr"]}, gamma={hypers["gamma"]}, tau={hypers["tau"]}, {hypers["action_noise"]} noise\n')

    # pe_fig, pe_axs = plt.subplots(3, 1, figsize=(20,9))

    for episode in range(hypers["num_epochs"]):
        epoch_start_time = time.time()
        state, _ = env.reset()        
        episode_reward = 0

        epoch_action_history = np.empty((0,2))
        epoch_reward_history = np.empty((0,))

        for step in range(hypers["num_steps"]):
            action, action_no_noise = agent.get_action(state, step)
            new_state, reward, done, _, _ = env.step(action, step)
            agent.replay_buffer.push(state, action, reward, new_state, done)
            
            if len(agent.replay_buffer) > hypers["batch_size"]:
                agent.update(hypers["batch_size"])
                    
            state = new_state
            episode_reward += reward

            if done:
                break
            
            epoch_action_history = np.append(epoch_action_history, np.array([action_no_noise]), axis=0)
            epoch_reward_history = np.append(epoch_reward_history, reward)

            # if episode % 10 == 0:
            #     plot_epoch(pe_fig, pe_axs, episode, step, epoch_action_history, epoch_reward_history, reward, state, env)
            
            
        rewards.append(episode_reward / hypers["num_steps"])
        avg_rewards.append(np.mean(rewards[-10:]))

        print(f'Episode {episode}: average reward: {round(rewards[-1], 3)}, in {round(time.time()-epoch_start_time, 3)}s\tgoal was {[round(value, 2) for value in state[-2:]]} {f"|Completion in {step} steps|" if done else ""}')

        # epoch_summary(episode, epoch_action_history, epoch_reward_history, rewards, action_bound)

    # compute and format training time
    total_time = time.time()-start_time
    (hours, minutes), seconds  = divmod(divmod(total_time, 60)[0], 60), (divmod(total_time, 60)[1])
    print(f'\nTotal time for {hypers["num_epochs"]} epochs: {int(hours)}h {int(minutes)}m {round(seconds, 2)}s')

    # eval runs
    print("\n")
    num_evals = 1000
    eval_reward = 0
    eval_start_time = time.time()
    for run in range(num_evals):
        print(f"Eval run {run}: ", end = "")
        eval_reward += eval_run(agent, env, hypers)

    eval_performance = eval_reward / num_evals # average avg dist to goal across evaluation runs
    print(f'Average reward over {num_evals} eval runs: {eval_performance} in {round(time.time()-eval_start_time, 2)} s')

    # save model and additional information if needed
    fig = end_plots(agent, rewards, avg_rewards, hypers)
    if save:    
        if models_path is not None:
            agent.save(models_path, rewards, avg_rewards, fig, eval_performance)
        else:
            print("Cant save models, no path provided")

if __name__ == "__main__":

    hypers = {"num_epochs": 1000,
                "batch_size": 100,
                "policy_lr": 1e-6,
                "critic_lr": 0.01,
                "gamma": 0.9,
                "tau": 0.005,
                "action_noise": "G",
                "g_noise_std": 0.02,
                "replay_buffer_size": int(1e6),
                "hidden_units": 512, # not currently used
                "num_steps": 500,
                "configuration": [('R', 10), ('R', 10)],
                }
    
    train_loop(hypers, models_path="models", save=True)