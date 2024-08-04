import torch
from torch import nn
import torch.nn.functional as F

from helpers import *
from plotting import *

from dqn_4_by_512 import DQN_4_by_512 as DQN_flappy

class AdaptedModel(nn.Module):
    def __init__(self, original_model, new_observations, new_actions, hidden_dim=512):
        
        super(AdaptedModel, self).__init__()
        # Extract all layers except the final layer
        self.features = nn.ModuleList(*list(original_model.children())[:-1])
        from_new = nn.Linear(new_observations, hidden_dim)
        self.features = nn.ModuleList(from_new, *self.features)
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

if __name__ == '__main__':

    device = 'cpu'
    num_states = 4
    num_actions = 2

    path_to_trained_model = get_path_to_trained_model(initialdir='trained_model_4x512_2')
    state_dict = torch.load(path_to_trained_model)
    flappy_model = DQN_flappy(state_size=12, action_size=2, hidden_dim=512).to(device)
    flappy_model.load_state_dict(state_dict)

    policy_dqn = AdaptedModel(original_model=flappy_model, new_observations=num_states, new_actions=num_actions).to(device)
        