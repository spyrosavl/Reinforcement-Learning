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
from torch.distributions import Categorical
import matplotlib.pyplot as plt

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(4, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        action_scores = self.linear1(x)
        return F.softmax(action_scores, dim=1)

def select_action(policy, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    actions_distribution = Categorical(probs)
    action = actions_distribution.sample()
    policy.saved_log_probs.append(actions_distribution.log_prob(action))
    return action.item()

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
    for log_prob, Gt in zip(policy.saved_log_probs, future_returns): # Use trajectory to estimate the policy loss
        policy_loss_list.append(-log_prob * Gt) # policy loss is the negative log probability of the action times the discounted return
    return torch.cat(policy_loss_list).sum() # sum up gradients

def fd_for_one_parameter(env, policy, dim1, dim2, no_of_pertubations=2, epsilon=np.finfo(np.float32).eps.item(), gamma=0.9):
    epsilon = epsilon * gamma
    #sample perturbations uniformly from [-epsilon, epsilon]
    perdubations = (torch.rand(no_of_pertubations) * 2 * epsilon - epsilon).tolist()
    policy_losses = []
    with torch.no_grad():
        for perdubation in perdubations:
            perdubated_policy = Policy() # create a new policy
            perdubated_policy.linear1.weight.data[dim1][dim2] = policy.linear1.weight.data[dim1][dim2] + perdubation
            sample_episode(env, perdubated_policy)
            policy_loss = calculate_policy_loss(perdubated_policy)
            policy_losses.append(policy_loss)
        policy_loss_gradient = torch.tensor(policy_losses).sum() / torch.tensor([abs(perdubation) for perdubation in perdubations]).sum()
    return policy_loss_gradient

def update_policy_reinforce(env, policy, optimizer):
    policy_loss = calculate_policy_loss(policy)
    optimizer.zero_grad() # clear gradients
    policy_loss.backward() # backpropagate
    optimizer.step() # update policy
    del policy.rewards[:] # clear rewards
    del policy.saved_log_probs[:] # clear log_probs

def update_policy_fd(env, policy, episode, gamma=0.9):
    for dim1 in range(policy.linear1.weight.data.shape[0]):
        for dim2 in range(policy.linear1.weight.data.shape[1]):
            policy_loss_gradient = fd_for_one_parameter(env, policy, dim1, dim2, gamma=gamma**episode)
            policy.linear1.weight.data[dim1][dim2] += - 1e-2 * policy_loss_gradient
    #clear rewards
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def sample_episode(env, policy):
    state, ep_reward = env.reset(), 0
    for t in range(1, 10000): # Don't infinite loop while learning
        action = select_action(policy, state)
        state, reward, done, _ = env.step(action)
        if args.render:
            env.render()
        policy.rewards.append(reward)
        ep_reward += reward
        if done:
            break
    return t, ep_reward

def main(seed, number_of_episodes=200):
    env = gym.make(args.env)
    env.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2) #only used for reinforce
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
    parser.add_argument('--method', required=True,
                        help='gradients calculation method (reinforce or fd)')
    parser.add_argument('--env', default="CartPole-v0", 
                        help='name of the environment to run')
    parser.add_argument('--no_of_episodes', type=int, default=200, metavar='N',
                        help='number of episodes to run expirements for')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    args = parser.parse_args()
    
    rewards = []
    #run main for 5 seeds
    for seed in range(2):
        rewards.append(main(seed, args.no_of_episodes))
    #plot the rewards
    plt.plot(torch.arange(1, args.no_of_episodes+1), torch.tensor(rewards).mean(dim=0))
    plt.title('Average rewards')
    plt.show()