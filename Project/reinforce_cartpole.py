#Algorithm from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
#REINFORCE is a gradient-based algorithm that uses Monte Carlo sampling to approximate the gradient of the policy.
#REINFORCE explained: https://towardsdatascience.com/policy-gradient-methods-104c783251e0
import argparse
import gym
import numpy as np
from itertools import count
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


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


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

class Linear_epsilon(nn.Module):

    __constants__ = ['in_features', 'out_features', 'mode']
    in_features: int
    out_features: int
    mode: str
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, mode: str, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear_epsilon, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.epsilon = torch.ones(self.out_features, self.in_features) * 0.1
        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.mode == 'add':
            nn.init.kaiming_uniform_(self.weight+self.epsilon, a=math.sqrt(5))
        elif self.mode == 'subtract':
            nn.init.kaiming_uniform_(self.weight-self.epsilon, a=math.sqrt(5))
        else:
            raise ValueError('Please try a different mode')
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Policy(nn.Module):
    def __init__(self, gamma, epsilon, batch_size):
        super(Policy, self).__init__()

        self.fc = nn.Linear(4, 2)
        self.fc_e_add = Linear_epsilon(4, 2, mode='add')
        self.fc_e_subtract = Linear_epsilon(4, 2, mode='subtract')

        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size

        self.saved_log_probs = []
        self.saved_log_probs_add = []
        self.saved_log_probs_sub = []

        self.rewards = []
        self.rewards_epsilon_add = []
        self.rewards_epsilon_sub = []

    def forward(self, x):
        force = self.fc(x)
        return F.softmax(force, dim=1)
    
    def forward_epsilon_add(self, x):
        force = self.fc_e_add(x)
        return F.softmax(force, dim=1)
    
    def forward_epsilon_subtract(self, x):
        force = self.fc_e_subtract(x)
        return F.softmax(force, dim=1)


policy = Policy(0.99, 0.1, 1)
optimizer = optim.SGD(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def force(state, mode):
    state = torch.from_numpy(state).float().unsqueeze(0)
    if mode == 'reinforce':
        probs = policy.forward(state)
    elif mode == 'finite_add':
        probs = policy.forward_epsilon_add(state)
    elif mode == 'finite_subtract':
        probs = policy.forward_epsilon_subtract(state)
    else:
        raise ValueError('Please try a different mode')

    actions_distribution = Categorical(probs)
    action = actions_distribution.sample()

    policy.saved_log_probs.append(actions_distribution.log_prob(action))
    policy.saved_log_probs_add.append(actions_distribution.log_prob(action))
    policy.saved_log_probs_sub.append(actions_distribution.log_prob(action))

    return action.item()


def update_policy_reinforce():
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
    optimizer.zero_grad() # clear gradients
    policy_loss = torch.cat(policy_loss_list).sum() # sum up gradients
    policy_loss.backward() # backpropagate
    optimizer.step() # update policy
    del policy.rewards[:] # clear rewards
    del policy.saved_log_probs[:] # clear log_probs


def update_policy_finite():

    R = 0
    policy_loss_list_1 = [] 
    future_returns_add = []
    for r_add in policy.rewards_epsilon_add[::-1]: # reverse buffer r
        R = r_add + args.gamma * R # G_t = r_t + gamma*G_{t+1}
        future_returns_add.insert(0, R) # insert at the beginning
    future_returns_add = torch.tensor(future_returns_add)
    future_returns_add = (future_returns_add - future_returns_add.mean()) / (future_returns_add.std() + eps) # normalize returns
    for log_prob_add, Gt in zip(policy.saved_log_probs_add, future_returns_add): # Use trajectory to estimate the policy loss
        policy_loss_list_1.append(-log_prob_add * Gt) # policy loss is the negative log probability of the action times the discounted return
    
    R = 0
    policy_loss_list_2 = [] 
    future_returns_sub = []
    for r_sub in policy.rewards_epsilon_sub[::-1]: # reverse buffer r
        R = r_sub + args.gamma * R # G_t = r_t + gamma*G_{t+1}
        future_returns_sub.insert(0, R) # insert at the beginning
    future_returns_sub = torch.tensor(future_returns_sub)
    future_returns_sub = (future_returns_sub - future_returns_sub.mean()) / (future_returns_sub.std() + eps) # normalize returns
    for log_prob_sub, Gt in zip(policy.saved_log_probs_sub, future_returns_sub): # Use trajectory to estimate the policy loss
        policy_loss_list_2.append(-log_prob_sub * Gt) # policy loss is the negative log probability of the action times the discounted return
    
    optimizer.zero_grad() # clear gradients
    policy_loss_1 = torch.cat(policy_loss_list_1).sum() # sum up gradients
    policy_loss_2 = torch.cat(policy_loss_list_2).sum() # sum up gradients
    policy_loss = (policy_loss_1 + policy_loss_2) / 2 
    policy_loss.backward() # backpropagate
    optimizer.step() # update policy
    del policy.rewards_epsilon_add[:] # clear rewards
    del policy.saved_log_probs_add[:] # clear log_probs
    del policy.rewards_epsilon_sub[:] # clear rewards
    del policy.saved_log_probs_sub[:] # clear log_probs


def sample_episode_reinforce():
    state, ep_reward = env.reset(), 0
    for t in range(1, 10000): # Don't infinite loop while learning
        action = force(state, 'reinforce')
        state, reward, done, _ = env.step(action)
        if args.render:
            env.render()
        policy.rewards.append(reward)
        ep_reward += reward
        if done:
            break
    return t, ep_reward

def sample_episode_finite_add():
    state_add, ep_reward_add = env.reset(), 0
    for t_add in range(1, 10000): # Don't infinite loop while learning
        action_add = force(state_add, 'finite_add')
        state_add, reward_add, done, _ = env.step(action_add)
        if args.render:
            env.render()
        policy.rewards_epsilon_add.append(reward_add)
        ep_reward_add += reward_add
        if done:
            break
    return t_add, ep_reward_add
   

def sample_episode_finite_sub():
    state_sub, ep_reward_sub = env.reset(), 0
    for t_sub in range(1, 10000): # Don't infinite loop while learning
        action_sub = force(state_sub, 'finite_subtract')
        state_sub, reward_sub, done, _ = env.step(action_sub)
        if args.render:
            env.render()
        policy.rewards_epsilon_sub.append(reward_sub)
        ep_reward_sub += reward_sub
        if done:
            break
    return t_sub, ep_reward_sub

def run_finite():
    running_reward_add = 10
    running_reward_sub = 10

    for i_episode in count(1):
        state_add, ep_reward_add = env.reset(), 0
        state_sub, ep_reward_sub = env.reset(), 0

        #sample an episode
        t_sub, ep_reward_sub = sample_episode_finite_sub()
        t_add, ep_reward_add = sample_episode_finite_add()

        # running reward across episodes
        running_reward_add = 0.05 * ep_reward_add + (1 - 0.05) * running_reward_add
        running_reward_sub = 0.05 * ep_reward_sub + (1 - 0.05) * running_reward_sub

        # update policy
        update_policy_finite()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward add: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, (ep_reward_add+ep_reward_sub)/2, (running_reward_add+running_reward_sub)/2))
        if (running_reward_add+running_reward_sub)/2 > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format((running_reward_add+running_reward_sub)/2, (t_add+t_sub)/2))
            break
        
def run_reinforce():
    running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        #sample an episode
        last_episode_final_step, ep_reward = sample_episode_reinforce()
        # running reward across episodes
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        # update policy
        update_policy_reinforce()

        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, last_episode_final_step))
            break
    
if __name__ == '__main__':
    # run_finite()
    run_reinforce()