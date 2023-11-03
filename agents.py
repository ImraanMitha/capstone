import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from networks import *
from utils import *


# agent consists of four networks, critic/actor and main/target
class DDPGagent:
    def __init__(self, env, hidden_size=512, actor_learning_rate=1e-4, critic_learning_rate=1e-4, gamma=0.5, tau=1e-1, noise_std = 0.1*np.pi, replay_buffer_size=50000):
        # Params
        self.num_actions = len(env.configuration)
        self.num_states = len(env.state)
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)

        # changed critic outputs from num_actions to 1
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, 1)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, 1)

        # this seems like a stupid way to do this, surely theres a better way
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.replay_buffer = ReplayBuffer(replay_buffer_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
    
    def get_action(self, state, add_noise = False):
        # state = torch.from_numpy(state).float().unsqueeze(0).requires_grad_() # why requires grad here
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = self.actor.forward(state)
        action = action.squeeze().detach().numpy()
        if add_noise:
            noise = np.random.normal(0, self.noise_std, action.shape)
            action = action + noise 

        action = np.clip(action, -np.pi, np.pi)
        return action
    
    
    def update(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size) #dont use done batch rn
        # also why am I using FloatTensor instead of just tensor?
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
    
        # Critic loss        
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q # yi in paper I thik 
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        # with this call to critic.forward, does it not extend the computational graph for parameters of critic? we dont want
        # this operation to be included in the critic gradient right?
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        
        # update networks
        # I think if my above assumption is true then after policy_loss.backwards and optim.step the critic parameters should change
        #TODO: So I could check the values before and after and see
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
