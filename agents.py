import os
import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from networks import *
from utils import *


# agent consists of four networks, critic/actor and main/target
#TODO: implement loading networks
class DDPGagent:
    def __init__(self, env, device, hidden_size=512, actor_learning_rate=1e-4, critic_learning_rate=1e-4, gamma=0.5, tau=1e-1, g_noise_std = 0.1*np.pi, replay_buffer_size=50000):
        # Params
        self.num_actions = len(env.configuration)
        self.num_states = len(env.state)
        self.gamma = gamma
        self.tau = tau
        self.gaussian_noise_std = g_noise_std
        self.ou_noise = OUNoise(self.num_actions)
        self.device = device

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions).to(self.device)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions).to(self.device)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, 1).to(self.device)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, 1).to(self.device)

        # copy target networks state dicts
        # this seems like a stupid way to do this, surely theres a better way
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training tools
        self.replay_buffer = ReplayBuffer(replay_buffer_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        # Loss histories
        self.policy_loss_history = [] # track policy loss, can delete later
        self.critic_loss_history = [] # track critic loss, can delete later


    def get_action(self, state, step, add_noise = None):
        #TODO: does actor need to be no_grad here?

        self.actor.eval()

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.actor(state)
        action = action.cpu().squeeze().detach().numpy()

        self.actor.train()

        action_no_noise = np.copy(action) # just for a test I am running delete this later

        if add_noise == 'G':
            noise = np.random.normal(0, self.gaussian_noise_std, action.shape)
            action = action + noise 
        elif add_noise == "OU":
            action = self.ou_noise.add_noise(action, step)

        action = np.clip(action, -0.1, 0.1) #TODO: make the action bounds class variables for agent, used here, in actor network definition, and in plotting in main to viz actions
        return action, action_no_noise
    
    
    def update(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size) #dont use done batch rn. this will be used as (1-done) in yi but I dont get any done=True so far so it doesnt really matter yet
            
        # using floattensor to be explicit about datatype when doing conversion from numpys standard float64 to torch float32
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
    
        ### Critic loss ###
        # target Q values (based on target network)
        target_Q = self.critic_target(next_states, self.actor_target(next_states))
        target_Q = rewards + (self.gamma * target_Q).detach() # still not fully sure why detach here

        # current Q estimate based on Q network
        current_Q = self.critic(states, actions)

        # critic loss (MSE)
        # critic_loss = self.critic_criterion(current_Q, target_Q)
        critic_loss = nn.functional.mse_loss(current_Q, target_Q)

        
        # optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()
        
        # #####old#####
        # Qvals = self.critic.forward(states, actions)
        # next_actions = self.actor_target.forward(next_states)
        # # next_Q = self.critic_target.forward(next_states, next_actions.detach()) # why detach here?
        # next_Q = self.critic_target.forward(next_states, next_actions)
        # Qprime = rewards + self.gamma * next_Q # yi in paper
        # critic_loss = self.critic_criterion(Qvals, Qprime)

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
    saves all 4 models under ./models/ directory in increminting subdirectories
    '''
    def save_models(self):
        models_path = 'models'
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
        #TODO: finish implementing, I will change hyperparams to be passed as a dict so we can have all of them here and keep the code clean but I first want to make sure I didnt mess anything up so far and push the changes
        parameter_file = os.path.join(next_dir_path, "parameters.txt")
        with open(parameter_file, 'w') as f:
            f.write()

