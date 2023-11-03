import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
# import torch.autograd

# TODO: right now these networks are 2 hidden layers with the same dimensions, this should be changed later
# TODO: should use nn.sequential instead of this implementation of forward

# Q value estimator networks
class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Expects state and action as torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


# policy networks
class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        """
        Expects state and action as torch tensors
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = np.pi * torch.tanh(self.linear3(x))

        return x