import numpy as np
import matplotlib.pyplot as plt
from agents import *
from utils import *
from manipulator_environment import *
import time
import gym

'''
Trains over one arm configuration.
'''
def train_loop(hypers, agent, env):
    print_period = 100 # how many episodes between printing episode information
    num_actions = env.action_dim
    
    rewards = []
    avg_rewards = []
    
    config_start_time = time.time()
    batch_start_time = time.time()

    # fig, axs for plotting episode if necessary
    pe_fig, pe_axs = plt.subplots(num_actions, 1, figsize=(20,9))

    # iterate through episodes
    for episode in range(hypers["num_episodes"]):
        state, _ = env.reset()
        episode_return = 0

        epoch_action_history = np.empty((0,num_actions))
        epoch_reward_history = np.empty((0,))

        # iterate through steps
        for step in range(hypers["num_steps"]):
            # get action, step environment and push experience 
            action, action_no_noise = agent.get_action(state, step)
            new_state, reward, done, _, _ = env.step(action, step)
            agent.replay_buffer.push(state, action, reward, new_state, done)
            
            # start doing updates every step once buffer has accumulated enough experiences
            if len(agent.replay_buffer) > hypers["batch_size"]:
                agent.update(hypers["batch_size"])
               
            state = new_state
            episode_return += reward

            # if this step has taken arm to terminal status, end the episode
            if done:
                break
            
            # track action and rewards
            epoch_action_history = np.append(epoch_action_history, np.array([action_no_noise]), axis=0)
            epoch_reward_history = np.append(epoch_reward_history, reward)

            # plot_episode(pe_fig, pe_axs, episode, step, epoch_action_history, epoch_reward_history, reward, state, env)
        
        # epoch_summary(episode, epoch_action_history, epoch_reward_history, rewards, env.action_bound)
        
        # compute return 
        rewards.append(episode_return / hypers["num_steps"])
        avg_rewards.append(np.mean(rewards[-10:]))

        # print information about the episode
        if (episode+1) % print_period == 0:
            print(f'\tEpisodes {episode-(print_period-1)}-{episode}: Normalized average reward: {round(np.mean(rewards[-print_period:])/env.working_radius, 3)} in {round(time.time()-batch_start_time, 3)} s')
            batch_start_time = time.time()

    '''
    Agent evaluation
    '''
    eval_reward = 0
    eval_start_time = time.time()
    for run in range(hypers["num_evals"]):
        eval_reward += eval_run(run, agent, env, hypers, verbose=False, plot=False)

    eval_performance = eval_reward / hypers["num_evals"] # average avg dist to goal across evaluation runs
    print(f'\n\tConfiguration normalized evaluation performance: {round(eval_performance, 4)} in {round(time.time()-eval_start_time, 2)} s')
    
    # Compute and format config time
    total_time = time.time()-config_start_time
    (hours, minutes), seconds  = divmod(divmod(total_time, 60)[0], 60), (divmod(total_time, 60)[1])
    print(f'\tTime for config: {int(hours)}h {int(minutes)}m {round(seconds, 2)}s\n')
    
    return rewards, avg_rewards

'''
Meta-training loop to train an agent over many environments and evaluate its generalized performance
'''
def meta_train(hypers, models_path=None, save=True):
    main_start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Beginning training on {device}')
    print(f'\n{hypers["num_configs"]} configurations, {hypers["num_episodes"]} episodes/config, {hypers["num_steps"]} steps/episode, {hypers["batch_size"]} batch size, lr_p={hypers["policy_lr"]}, lr_c={hypers["critic_lr"]}, gamma={hypers["gamma"]}, tau={hypers["tau"]}, {hypers["action_noise"]} noise\n')

    rewards = []
    avg_rewards = []
    
    # dummy environment to set certain parameters
    dummy_env =  Planar_Environment(configuration=[('R', 1)]*hypers['num_joints'])
    num_actions = dummy_env.action_dim
    num_states = dummy_env.state_dim
    action_bound = dummy_env.action_bound
    agent = DDPGagent(action_bound, num_actions, num_states, device, hypers)

    # loop over configurations to train on
    for config_num in range(hypers['num_configs']):
        config = [('R', length) for length in np.random.uniform(hypers['config_low'], hypers['config_high'], hypers['num_joints'])]
        print(f'Configuration {config_num}: {[round(joint[1], 2) for joint in config]}')

        # environment setup
        env = Planar_Environment(configuration=config)
        
        # train on the given configuration
        conf_rewards, conf_avg_rewards = train_loop(hypers, agent, env)
        rewards.extend(conf_rewards)
        avg_rewards.extend(conf_avg_rewards)

        # General config evaluation (run every config)
        eval_reward = 0
        eval_start_time = time.time()
        for run in range(hypers["num_evals"]):
            config = [('R', length) for length in np.random.uniform(hypers['config_low'], hypers['config_high'], hypers['num_joints'])]
            env = Planar_Environment(configuration=config)
            eval_reward += eval_run(run, agent, env, hypers, verbose=False, plot=False)

        eval_performance = eval_reward / hypers["num_evals"] # average avg dist to goal across evaluation runs
        print(f'\tAgent normalized evaluation performance: {round(eval_performance, 4)} in {round(time.time()-eval_start_time, 2)} s\n')


    '''
    Final general config evaluation
    '''
    eval_reward = 0
    eval_start_time = time.time()
    for run in range(hypers["num_evals"]):
        config = [('R', length) for length in np.random.uniform(hypers['config_low'], hypers['config_high'], hypers['num_joints'])]
        env = Planar_Environment(configuration=config)
        eval_reward += eval_run(run, agent, env, hypers, verbose=False, plot=False)

    eval_performance = eval_reward / hypers["num_evals"] # average normalized avg dist to goal across evaluation runs
    print(f'Final agent normalized evaluation performance: {round(eval_performance, 4)} in {round(time.time()-eval_start_time, 2)} s')

    # Compute and format total run time
    total_time = time.time()-main_start_time
    (hours, minutes), seconds  = divmod(divmod(total_time, 60)[0], 60), (divmod(total_time, 60)[1])
    print(f'Total time: {int(hours)}h {int(minutes)}m {round(seconds, 2)}s\n')
    
    '''
    Save model and additional information if requested.
    '''
    fig = end_plots(agent, rewards, avg_rewards, hypers)
    if save:    
        if models_path is not None:
            agent.save(models_path, rewards, avg_rewards, fig, eval_performance)
        else:
            print("Cant save models, no path provided")


if __name__ == "__main__":

    hypers = {"num_configs": 10,
              "num_episodes": 1000,
                "batch_size": 100,
                "policy_lr": 1e-6,
                "critic_lr": 0.01,
                "gamma": 0.9,
                "tau": 0.005,
                "action_noise": "G",
                "g_noise_std": 0.02,
                "replay_buffer_size": int(1e8),
                "hidden_units": 100,
                "num_steps": 250,
                "config_low": 2,
                "config_high": 10,
                "num_joints": 3,
                "num_evals": 1000,
                }
    
    meta_train(hypers, models_path="models", save=True)