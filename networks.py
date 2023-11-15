import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

# TODO: right now these networks are 2 hidden layers with the same dimensions, this should be changed later
# TODO: should use nn.sequential instead of this implementation of forward

# Q value estimator networks
# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_size):
#         super(Critic, self).__init__()
#         self.bn0 = nn.BatchNorm1d(state_dim + action_dim)
#         self.linear1 = nn.Linear(state_dim + action_dim, hidden_size)
#         self.bn1 = nn.BatchNorm1d(hidden_size)
#         self.linear2 = nn.Linear(hidden_size, hidden_size)
#         self.bn2 = nn.BatchNorm1d(hidden_size)
#         self.linear3 = nn.Linear(hidden_size, 1)

#     def forward(self, state, action):
#         x = torch.cat([state, action], 1)
#         x = self.bn0(x)
#         x = F.relu(self.linear1(x))
#         x = self.bn1(x)
#         x = F.relu(self.linear2(x))
#         x = self.bn2(x)
#         x = self.linear3(x)

#         return x


# # policy networks
# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_size, action_bound):
#         super(Actor, self).__init__()
#         self.action_bound = float(action_bound[0])
#         self.bn0 = nn.BatchNorm1d(state_dim)
#         self.linear1 = nn.Linear(state_dim, hidden_size)
#         self.bn1 = nn.BatchNorm1d(hidden_size)
#         self.linear2 = nn.Linear(hidden_size, hidden_size)
#         self.bn2 = nn.BatchNorm1d(hidden_size)
#         self.linear3 = nn.Linear(hidden_size, action_dim)
        
#     def forward(self, state):
#         x = self.bn0(state)
#         x = F.relu(self.linear1(x))
#         x = self.bn1(x)
#         x = F.relu(self.linear2(x))
#         x = self.bn2(x)
#         x = self.action_bound*torch.tanh(self.linear3(x))

#         return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, action_bound):
        super(Actor, self).__init__()
        max_action = float(action_bound[0])
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 400)
        self.l3 = nn.Linear(400, 300)
        self.l4 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.max_action * torch.tanh(self.l4(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400 , 400)
        self.l3 = nn.Linear(400, 300)
        self.l4 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, action_bound):
        super(Actor, self).__init__()
        max_action = float(action_bound[0])
        self.l1 = nn.Linear(state_dim, 100)
        self.l2 = nn.Linear(100, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.max_action * torch.tanh(self.l2(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 100)
        self.l2 = nn.Linear(100, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = self.l2(x)
        return x
