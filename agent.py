import itertools
import torch
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
import yaml
import random
from plotting import *

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
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']

        self.loss = torch.nn.MSELoss()
        self.optimizer = None

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
            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            training_steps = 0

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        
        rewards_per_episode = []
        epsilon_history = []
        durations_per_episode = []
        losses = []

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

                if terminated:
                    reward = -300

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
            # plot_durations(durations_per_episode)
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            epsilon_history.append(epsilon)

            if is_training and len(memory) >= self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                loss = self.optimize(mini_batch, policy_dqn, target_dqn)
                losses.append(loss)
                plot_losses(losses)


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
    agent = Agent('cartpole1')
    agent.run(is_training=True, render=False)
