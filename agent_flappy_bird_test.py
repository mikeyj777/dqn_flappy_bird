import os
import itertools
import torch
import flappy_bird_gymnasium
import gymnasium
import yaml
import random

from datetime import datetime, timedelta
import argparse



from dqn import DQN
from experience_replay import ReplayMemory
from helpers import *
from plotting import *

seed = 44

random.seed(seed)

game = 'flappybird1'
train = False

DATE_FORMAT = "%Y_%m_%d_%H%M%S"
DATE_TIME_STAMP = f'{game}_{datetime.now().strftime(DATE_FORMAT)}'

RUNS_DIR = f'runs/{DATE_TIME_STAMP}'
os.makedirs(RUNS_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

class Agent:

    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            print(hyperparameters)

        self.hyperparameter_set = hyperparameter_set
        self.env_id = hyperparameters['env_id']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.env_make_params = hyperparameters.get('env_make_params', {}) # try to get the params.  if key not there, return empty dict
        self.stop_on_reward = hyperparameters['stop_on_reward']

        self.loss = torch.nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True, render=False):
        
        # env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False).env
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time
            log_message = f'started at {start_time.strftime(DATE_FORMAT)}:  Training starting...'
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')
            
        
        env = gymnasium.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]

        rewards_per_episode = []
        epsilon_history = []
        durations_per_episode = []
        losses = []
        
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
        
        epsilon = self.epsilon_init

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            training_steps = 0

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            best_reward = -9999999
        else:

            filenm = get_path_to_trained_model()
            policy_dqn.load_state_dict(torch.load(filenm))

            #switch to evaluation mode
            policy_dqn.eval()
        
        for episode in itertools.count():
            episode_reward = 0.0
            state, _ = env.reset(seed=seed)
            state = torch.tensor(state, device=device, dtype=torch.float32)
            terminated = False
            duration = 0
            while not terminated and episode_reward < self.stop_on_reward:
                duration += 1
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, device=device, dtype=torch.int64)
                else:   
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                if duration == 50:
                    apple = 1

                # Processing:
                new_state, reward, terminated, truncated, info = env.step(action.item())

                if terminated:
                    # print(f'{duration = }')
                    apple = 1
                    # reward = -300

                new_state = torch.tensor(new_state, device=device, dtype=torch.float32)
                reward = torch.tensor(reward, device=device, dtype=torch.float32)
                
                episode_reward += reward
                if is_training:
                    memory.append((state, action, new_state, reward, terminated))

                    training_steps += 1
                    # print(training_steps)

                state = new_state
            
            rewards_per_episode.append(episode_reward)
            durations_per_episode.append(duration)
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            epsilon_history.append(epsilon)

            if not is_training:
                continue
            
            plot_param(rewards_per_episode)

            if episode_reward.item() > best_reward:
                    training_time = datetime.now() - start_time
                    log_message = f'{datetime.now().strftime(DATE_FORMAT)} | duration: {training_time} | num_steps:  {duration} | new best reward: {episode_reward:0.1f} | percent improved: {(100 * (best_reward - episode_reward) / best_reward):0.1f}%'
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                    
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward.item()
                
            # update graph
            current_time = datetime.now()
            if current_time - last_graph_update_time > timedelta(seconds=10):
                if len(rewards_per_episode) == 0:
                    rewards_per_episode.append(episode_reward)
                # save_or_show_graph(rewards_per_episode, epsilon_history, durations_per_episode, losses, self.GRAPH_FILE, save_fig=False)
                last_graph_update_time = current_time
            
            if len(memory) >= self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                loss = self.optimize(mini_batch, policy_dqn, target_dqn)
                losses.append(loss)
                # plot_losses(losses)


                if training_steps > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    training_steps = 0
                    
                
        env.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        states, actions, new_states, rewards, terminateds = zip(*mini_batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminateds).float().to(device)

        with torch.no_grad():
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        
        current_q = policy_dqn(states).gather(dim = 1, index=actions.unsqueeze(dim=1)).squeeze()
        loss = self.loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description='Train or test model')
    # parser.add_argument('hyperparameters', help='')
    # parser.add_argument('--train', help='training mode', action='store_true')
    # args = parser.parse_args()

    dql = Agent(game)

    is_training=True
    render = False
    if not train:
        is_training = False
        render = True

    dql.run(is_training=is_training, render=render)

apple = 1