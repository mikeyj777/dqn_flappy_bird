import torch
from torch import nn
import torch.nn.functional as F

from helpers import *
from plotting import *

class AdaptedModel(nn.Module):
    def __init__(self, original_model, new_observations, new_actions, hidden_dim=512):
        
        super(AdaptedModel, self).__init__()
        # Extract all layers except the final layer
        self.features = nn.ModuleList([
            original_model.fc1,
            original_model.fc2,
            original_model.fc3,
        ])
        from_new = nn.Linear(new_observations, hidden_dim)
        self.features = nn.ModuleList([from_new, *self.features])
        # Add a new fully connected layer to match CartPole's action space
        self.fc = nn.Linear(self.features[-1].out_features, 2)  # CartPole has 2 actions: left or right
        
        # super(AdaptedModel, self).__init__()
        # # Extract all layers except the final layer
        # orig_models = list(original_model.children())
        # orig_in = orig_models[0]
        
        # # self.features = nn.Sequential(*orig_models[:-1])
        # models = []
        # models.append(nn.Linear(new_observations, hidden_dim))
        # # *list(original_model.children())[:-1]
        # self.features = nn.Sequential()
        # # Add a new fully connected layer to match CartPole's action space
        self.fc = nn.Linear(self.features[-1].out_features, 2)  # CartPole has 2 actions: left or right
    
    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x
