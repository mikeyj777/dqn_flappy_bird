import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

def plot_param(param):
    plt.ion()
    plt.figure(1)
    param = torch.tensor(param, dtype=torch.float)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.plot(param.numpy())
    # Take 100 episode averages and plot them too
    N = len(param)
    param_np = np.array(param)
    running_ave = np.empty(N)
    for t in range(N):
        running_ave[t] = param_np[max(0, t-100):(t+1)].mean()
    plt.plot(running_ave)

    plt.pause(0.001)  # pause a bit so that plots are updated

    plt.gcf()

def plot_losses(losses):
    plt.ion()
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.plot(losses)
    N = len(losses)
    losses_np = np.array(losses)
    running_ave = np.empty(N)
    for t in range(N):
        running_ave[t] = losses_np[max(0, t-100):(t+1)].mean()
    plt.plot(running_ave)
    plt.pause(0.001)
    plt.gcf()

def save_or_show_graph(rewards_per_episode, epsilon_decay, durations_per_episode, losses, graph_file_name, save_fig=False):
    fig = plt.figure(3)
    if save_fig:
        matplotlib.use('Agg')
    else:
        plt.ion()
        plt.clf()
    means = np.zeros((4, len(rewards_per_episode)), dtype=np.float32)
    data = [rewards_per_episode, epsilon_decay, durations_per_episode, losses]
    for i in range(len(data)):
        for j in range(len(data[i])):
            means[i, j] = np.mean(data[i][max(0, j-100):(j+1)])
    
    labels = ['rewards_per_episode', 'epsilon_decay', 'durations_per_episode', 'losses']
    for i in range(len(data)):
        subplot_id = 221 + i
        plt.ylabel(labels[i])
        plt.subplot(subplot_id)
        plt.plot(data[i])
        plt.plot(means[i])

    fig.subplots_adjust(wspace=0.5, hspace=1.0)

    if save_fig:
        fig.savefig(graph_file_name)
        plt.close(fig)
    else:
        plt.pause(0.001)
        plt.gcf()