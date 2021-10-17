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

def fd_for_one_parameter(env, policy, dim1, dim2):
    eps = np.finfo(np.float32).eps.item()
    perdubations = [eps, -eps]
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

def update_policy(env, policy, _):
    for dim1 in range(policy.linear1.weight.data.shape[0]):
        for dim2 in range(policy.linear1.weight.data.shape[1]):
            policy_loss_gradient = fd_for_one_parameter(env, policy, dim1, dim2)
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

def main():
    env = gym.make('CartPole-v1')
    env.seed(args.seed)
    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        #sample an episode
        last_episode_final_step, ep_reward = sample_episode(env, policy)
        # running reward across episodes
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        # update policy
        update_policy(env, policy, optimizer)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, last_episode_final_step))
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    main()