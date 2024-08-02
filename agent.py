import itertools
import torch
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
import yaml
import random
from plotting import plot_durations

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:

    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            print(hyperparameters)

        self.env_id = hyperparameters['env_id']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']

    def run(self, is_training=True, render=False):
        
        # env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False).env
        render_mode = 'rgb_array'
        if render:
            render_mode = 'human'
        env = gymnasium.make("CartPole-v1", render_mode=render_mode).env
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]

        policy_dqn = DQN(num_states, num_actions).to(device)

        epsilon = self.epsilon_init

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
        
        rewards_per_episode = []
        epsilon_history = []
        durations_per_episode = []

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, device=device, dtype=torch.float32)
            terminated = False
            episode_reward = 0.0
            duration = 0
            while not terminated:
                duration += 1
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, device=device, dtype=torch.int64)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())

                new_state = torch.tensor(new_state, device=device, dtype=torch.float32)
                reward = torch.tensor(reward, device=device, dtype=torch.float32)
                
                episode_reward += reward
                if is_training:
                    memory.append((state, action, new_state, reward, terminated))

                state = new_state
            
            rewards_per_episode.append(episode_reward)
            durations_per_episode.append(duration)
            plot_durations(durations_per_episode)
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            epsilon_history.append(epsilon)


        env.close()

if __name__ == "__main__":
    agent = Agent('cartpole1')
    agent.run(is_training=True, render=False)
