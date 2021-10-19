#Algorithm from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
#REINFORCE is a gradient-based algorithm that uses Monte Carlo sampling to approximate the gradient of the policy.
#REINFORCE explained: https://towardsdatascience.com/policy-gradient-methods-104c783251e0
import argparse
import gym
import numpy as np
from itertools import count, permutations
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.distributions.uniform import Uniform
from environment import CartPolev0
import matplotlib.pyplot as plt


# class Reinforce_Policy(nn.Module):
#     def __init__(self):
#         super(Reinforce_Policy, self).__init__()
#         self.linear1 = nn.Linear(4, 2)

#         self.saved_log_probs = []
#         self.rewards = []

#     def forward(self, x):
#         action_scores = self.linear1(x)
#         return F.softmax(action_scores, dim=1)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(4, 1)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0)
        force = self.linear1(x)
        return force

def select_action(policy, state):
    # state = torch.from_numpy(state).float().unsqueeze(0)
    force_mean = policy(state)
    actions_distribution = Normal(force_mean, 0.1)
    force = actions_distribution.sample()
    policy.saved_log_probs.append(actions_distribution.log_prob(force))
    return force.item()

def calculate_policy_loss(policy):
    eps = np.finfo(np.float32).eps.item()
    R = 0
    policy_loss_list = [] 
    future_returns = []
    for r in policy.rewards[::-1]: # reverse buffer r
        R = r + args.gamma * R # G_t = r_t + gamma*G_{t+1}
        future_returns.insert(0, R) # insert at the beginning
    future_returns = torch.tensor(future_returns)
    future_returns = (future_returns - future_returns.mean()) / (future_returns.std() + eps) # normalize returns
    if args.method == 'reinforce':
        for log_prob, Gt in zip(policy.saved_log_probs, future_returns): # Use trajectory to estimate the policy loss
            policy_loss_list.append(-log_prob * Gt) # policy loss is the negative log probability of the action times the discounted return
    else:
        for Gt in future_returns:
            policy_loss_list.append(-Gt)
    # print(policy_loss_list)
    return torch.cat(policy_loss_list).sum() if args.method == 'reinforce' else torch.tensor(policy_loss_list).sum()# sum up gradients

# def fd(env, policy, no_of_parameters, no_of_pertubations=2, epsilon=0.1, gamma=0.9):
#     epsilon = epsilon * gamma
#     #sample perturbations uniformly from [-epsilon, epsilon]
#     pertubations = Uniform(-epsilon, epsilon).sample((no_of_parameters,))
#     policy_losses = []
#     with torch.no_grad():
#         for pertubation in pertubations:
#             perdubated_policy = Finite_Policy() # create a new policy
#             perdubated_policy.linear1.weight.data = policy.linear1.weight.data + pertubation
#             sample_episode(env, perdubated_policy)
#             policy_loss = calculate_policy_loss(perdubated_policy)
#             policy_losses.append(policy_loss)
#         policy_loss_gradient = torch.tensor(policy_losses).sum() / torch.tensor([abs(perdubation) for perdubation in pertubations]).sum()
#         # print(policy_loss_gradient)
#     return policy_loss_gradient

def update_policy_reinforce(env, policy, optimizer):
    policy_loss = calculate_policy_loss(policy)
    optimizer.zero_grad() # clear gradients
    policy_loss.backward() # backpropagate
    optimizer.step() # update policy
    del policy.rewards[:] # clear rewards
    del policy.saved_log_probs[:] # clear log_probs

def update_policy_fd(env, policy, episode, epsilon=0.1, no_of_parameters=4, no_of_pertubations=2, gamma=0.9):
    epsilon *= gamma
    gamma = gamma**episode
    pertubations = Uniform(-epsilon, epsilon).sample((no_of_parameters,))
    policy_losses = []
    with torch.no_grad():
        for pertubation in pertubations:
            policy_loss = calculate_policy_loss(policy)
            perdubated_policy = policy # create a new policy
            perdubated_policy.linear1.weight.data = policy.linear1.weight.data + pertubation
            sample_episode(env, perdubated_policy)
            pert_policy_loss = calculate_policy_loss(perdubated_policy)
            policy_losses.append(pert_policy_loss-policy_loss)
        policy_loss_gradient = torch.tensor(policy_losses).sum() / torch.tensor([abs(perdubation) for perdubation in pertubations]).sum()
        # print(policy_loss_gradient)
    policy.linear1.weight.data += - 1e-2 * policy_loss_gradient
    
    #clear rewards
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def sample_episode(env, policy):
    state, ep_reward = env.reset(), 0
    for t in range(1, 10000): # Don't infinite loop while learning
        force = select_action(policy, state)
        state, reward, done, _ = env.step(force)
        if args.render:
            env.render()
        policy.rewards.append(reward)
        ep_reward += reward
        if done:
            break
    return t, ep_reward

def main(seed, number_of_episodes=200):
    env = args.env
    env.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    policy = Policy()
    optimizer = optim.SGD(policy.parameters(), lr=1e-2) #only used for reinforce
    running_reward = 10
    episode_rewards = []
    for i_episode in range(number_of_episodes):
        state, ep_reward = env.reset(), 0
        #sample an episode
        last_episode_final_step, ep_reward = sample_episode(env, policy)
        episode_rewards.append(ep_reward)
        # running reward across episodes
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        # update policy
        if args.method == 'reinforce':
            update_policy_reinforce(env, policy, optimizer)
        elif args.method == 'fd':
            update_policy_fd(env, policy, i_episode)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
    return episode_rewards
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--method', type=str, default='reinforce',
                        help='gradients calculation method (reinforce or fd)')
    parser.add_argument('--env', type=object, default=CartPolev0(), 
                        help='name of the environment to run')
    parser.add_argument('--no_of_episodes', type=int, default=4000, metavar='N',
                        help='number of episodes to run expirements for')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='G',
                        help='discount factor (default: 0.9)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--no_seeds', type=int, default=10, metavar='N',
                        help='number of seeds (default: 10)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    args = parser.parse_args()
    

    rewards = []; means = []; stds = []
    #run main for 5 seeds
    for seed in range(args.no_seeds):
        rewards.append(main(seed, args.no_of_episodes))
    
    rewards = np.asarray(rewards)

    for i in range(rewards.shape[1]):
        means.append(np.asarray(rewards[:,i]).mean())
        stds.append(np.asarray(rewards[:,i]).std())

    #plot the rewards
    plt.plot(torch.arange(1, args.no_of_episodes+1), torch.tensor(rewards).float().mean(dim=0))
    plt.title('Average rewards')
    plt.show()

    # import pdb; pdb.set_trace()

    X = np.arange(rewards.shape[1])
    y_upper = np.array([means[i] + np.sqrt(stds[i]) for i in range(len(means))])
    y_lower = np.array([means[i] - np.sqrt(stds[i]) for i in range(len(means))])

    plt.figure(figsize=(20,10))

    plt.plot(X, means, '-r', label='Prediction mean', c='orange', alpha=0.8)

    plt.fill_between(X, y_upper,y_lower, label='Prediction std', facecolor='orange', alpha=0.3)

    plt.xlabel("Episode")
    # plt.xticks(range(len(X)), X)
    plt.ylabel("Average reward")
    plt.legend(loc='upper left')
    plt.show()

    # import pdb; pdb.set_trace()