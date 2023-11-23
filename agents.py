import os
import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from networks import *
from utils import *


class DDPGagent:
    def __init__(self, action_bound, num_actions, num_states, device, hypers):
        # Params
        self.hypers = hypers
        self.gamma = hypers["gamma"]
        self.tau = hypers["tau"]
        self.action_noise = hypers["action_noise"]
        self.gaussian_noise_std = hypers["g_noise_std"]
      
        self.num_actions = num_actions
        self.num_states = num_states
        
        self.ou_noise = OUNoise(self.num_actions)

        self.device = device

        self.action_bound = action_bound

        # Networks
        hidden_size = hypers["hidden_units"]
        self.actor = Actor(self.num_states,  self.num_actions, hidden_size, self.action_bound).to(self.device)
        self.actor_target = Actor(self.num_states, self.num_actions, hidden_size, self.action_bound).to(self.device)

        self.critic = Critic(self.num_states, self.num_actions, hidden_size).to(self.device)
        self.critic_target = Critic(self.num_states, self.num_actions, hidden_size).to(self.device)
        
        ####### test of weight initialization #######
        # for layer in self.actor.children():
        #     nn.init.uniform_(layer.weight, -0.01, 0.01)
        #     nn.init.zeros_(layer.bias)

        # for layer in self.critic.children():
        #     nn.init.uniform_(layer.weight, -0.01, 0.01)
        #     nn.init.zeros_(layer.bias)
        #############################################

        # copy target networks state dicts
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Training tools
        self.replay_buffer = ReplayBuffer(hypers["replay_buffer_size"])        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=hypers["policy_lr"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=hypers["critic_lr"])

        # Loss histories
        self.policy_loss_history = [] # track policy loss
        self.critic_loss_history = [] # track critic loss

    '''
    Load the moel state dicts from a previous run. Expects the path to a dir with the 4 .pth files
    '''
    def load(self, dir, device):
        self.actor.load_state_dict(torch.load(os.path.join(dir, 'actor.pth'), map_location=device))
        self.actor_target.load_state_dict(torch.load(os.path.join(dir, 'actor_target.pth'), map_location=device))
        self.critic.load_state_dict(torch.load(os.path.join(dir, 'critic.pth'), map_location=device))
        self.critic_target.load_state_dict(torch.load(os.path.join(dir, 'critic_target.pth'), map_location=device))
    

    '''
    Gets an action from the policy given the state, adds action noise if desired
    '''
    def get_action(self, state, step):
        self.actor.eval() # placed in eval mode for this step since network involves bn layers

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.actor(state)
        action = action.cpu().data.numpy().flatten()

        self.actor.train()

        action_no_noise = np.copy(action)

        if self.action_noise == 'G':
            noise = np.random.normal(0, self.gaussian_noise_std, action.shape)
            action = action + noise 
        elif self.action_noise == "OU": #not well implemented, sigmas are way too big
            action = self.ou_noise.add_noise(action, step)

        action = np.clip(action, -self.action_bound, self.action_bound)
        return action, action_no_noise

    '''
    updates the networks using batches sampled from the replay buffer. Based on standard ddpg theory
    '''
    def update(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
            
        # using floattensor to be explicit about datatype when doing conversion from numpys standard float64 to torch float32
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        dones = torch.FloatTensor(dones.reshape(-1, 1)).to(self.device) # should just have this convert dones directly and in target_Q do (1-done) 
    
        ### Critic loss ###
        # target Q values (based on target network)
        target_Q = self.critic_target(next_states, self.actor_target(next_states))
        target_Q = rewards + ((1-dones) * self.gamma * target_Q).detach() # still not fully sure why detach here

        # current Q estimate based on Q network
        current_Q = self.critic(states, actions)

        # critic loss (MSE)
        critic_loss = self.critic_criterion(current_Q, target_Q)
        # critic_loss = nn.functional.mse_loss(current_Q, target_Q)

        # optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # Actor loss
        policy_loss = -self.critic(states, self.actor(states)).mean()

        # optimize actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # for visualizing loss history
        self.critic_loss_history.append(critic_loss.cpu().detach().numpy())
        self.policy_loss_history.append(policy_loss.cpu().detach().numpy())

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    '''
    saves all 4 models, plot png, and loss arrays under ./models/ directory in increminting subdirectories each run
    '''
    def save(self, models_path, rewards, avg_rewards, fig, eval_performance):
        zero_fill = 4

        existing_subdirs = [d for d in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, d))]

        if existing_subdirs:
            next_subdir = str(max(int(d) for d in existing_subdirs) + 1).zfill(zero_fill)
        else:
            next_subdir = '0'.zfill(zero_fill)

        next_dir_path = os.path.join(models_path, next_subdir)
        os.mkdir(next_dir_path)

        # save networks
        torch.save(self.actor.state_dict(), os.path.join(next_dir_path, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(next_dir_path, 'critic.pth'))
        torch.save(self.actor_target.state_dict(), os.path.join(next_dir_path, 'actor_target.pth'))
        torch.save(self.critic_target.state_dict(), os.path.join(next_dir_path, 'critic_target.pth'))

        # save parameters to a txt file
        parameter_file = os.path.join(next_dir_path, "parameters.txt")
        with open(parameter_file, 'w') as f:
            f.write(f'Run {next_subdir}\n\n')
            for key, value in self.hypers.items():
                f.write(f"{key}:\t{value}\n")
            f.write(f'\nAverage evaluation reward: {eval_performance}')
        
        # save loss arrays
        np.save(os.path.join(next_dir_path, "policy_loss.npy"), np.array(self.policy_loss_history))
        np.save(os.path.join(next_dir_path, "critic_loss.npy"), np.array(self.critic_loss_history))
        np.save(os.path.join(next_dir_path, "rewards.npy"), np.array(rewards))
        np.save(os.path.join(next_dir_path, "avg_rewards.npy"), np.array(avg_rewards))

        # save plots png
        fig.savefig(os.path.join(next_dir_path, f"plots{next_subdir}.png"))


