import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
# import torch.autograd

# TODO: right now these networks are 2 hidden layers with the same dimensions, this should be changed later
# TODO: should use nn.sequential instead of this implementation of forward

# Q value estimator networks
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Critic, self).__init__()
        self.bn0 = nn.BatchNorm1d(state_dim + action_dim)
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.bn0(x)
        x = F.relu(self.linear1(x))
        x = self.bn1(x)
        x = F.relu(self.linear2(x))
        x = self.bn2(x)
        x = self.linear3(x)

        return x


# policy networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        self.bn0 = nn.BatchNorm1d(state_dim)
        self.linear1 = nn.Linear(state_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.linear3 = nn.Linear(hidden_size, action_dim)
        
    def forward(self, state):
        x = self.bn0(state)
        x = F.relu(self.linear1(x))
        x = self.bn1(x)
        x = F.relu(self.linear2(x))
        x = self.bn2(x)
        x = self.action_bound*torch.tanh(self.linear3(x))

        return x