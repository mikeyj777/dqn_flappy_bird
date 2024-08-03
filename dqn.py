import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, state_size, action_size, hidden_dim=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_size)
    

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
