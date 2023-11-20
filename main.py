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
    env = Planar_Environment(configuration=hypers['configuration'], step_cost=0)
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

        epoch_action_history = np.empty((0,num_actions))
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

    # Compute and format training time
    total_time = time.time()-start_time
    (hours, minutes), seconds  = divmod(divmod(total_time, 60)[0], 60), (divmod(total_time, 60)[1])
    print(f'\nTotal time for {hypers["num_epochs"]} epochs: {int(hours)}h {int(minutes)}m {round(seconds, 2)}s')

    '''
    Agent evaluation
    '''
    print("\n")
    num_evals = 1000
    eval_reward = 0
    eval_start_time = time.time()
    for run in range(num_evals):
        print(f"Eval run {run}: ", end = "")
        eval_reward += eval_run(agent, env, hypers, plot=False)

    eval_performance = eval_reward / num_evals # average avg dist to goal across evaluation runs
    print(f'Average reward over {num_evals} eval runs: {round(eval_performance, 4)} in {round(time.time()-eval_start_time, 2)} s')


    '''
    Save model and additional information if requested
    '''
    fig = end_plots(agent, rewards, avg_rewards, hypers)
    if save:    
        if models_path is not None:
            agent.save(models_path, rewards, avg_rewards, fig, eval_performance)
        else:
            print("Cant save models, no path provided")

if __name__ == "__main__":

    hypers = {"num_epochs": 2000,
                "batch_size": 100,
                "policy_lr": 1e-7,
                "critic_lr": 0.01,
                "gamma": 0.9,
                "tau": 0.005,
                "action_noise": "G",
                "g_noise_std": 0.02,
                "replay_buffer_size": int(1e6),
                "hidden_units": 100,
                "num_steps": 250,
                "configuration": [('R', 5), ('R', 10), ('R', 5)],
                }
    
    train_loop(hypers, models_path="models", save=True)