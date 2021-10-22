#Algorithm from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
#REINFORCE is a gradient-based algorithm that uses Monte Carlo sampling to approximate the gradient of the policy.
#REINFORCE explained: https://towardsdatascience.com/policy-gradient-methods-104c783251e0
import argparse, pickle
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

class Policy(nn.Module):
    """
    Policy theta. 
    -------------------
    Args:
    
    x : torch.tensor representing the state(s). 

    Returns:

    force_mean for the custom env 
    action_probs for any other env

    -------------------
    """
    def __init__(self):
        super(Policy, self).__init__()

        if args.env == 'CartPole-v0':
            self.input = 4
            self.output = 2
        elif args.env == 'LunarLander-v2':
            self.input = 8
            self.output = 3
        elif args.env == 'CartPole-v0-custom':
            self.input = 4
            self.output = 1
        else:
            raise ValueError('This environment is not supported in our experiments.')

        self.linear1 = nn.Linear(self.input, self.output)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        output = self.linear1(x)
        return output if args.env == 'CartPole-v0-custom' else F.softmax(output, dim=1)

def select_force(policy, state):
    """
    This function returns the force used in the cartpole env.\
    The force is sampled by a normal distribution N(force_mean, args.std),\
    where the force_mean is encoded by the policy network.
    -------------------
    Args:

    policy: class object 
    state : ndarray provided by the env.step

    Returns:

    force : scalar

    ------------------
    """
    state = torch.from_numpy(state).float().unsqueeze(0)
    force_mean = policy(state)
    actions_distribution = Normal(force_mean, args.std)
    force = actions_distribution.sample()
    policy.saved_log_probs.append(actions_distribution.log_prob(force))
    return force.item()

def select_action(policy, state):
    """
    This function returns the action used in the cartpole env.\
    The action is sampled by a categorical distribution,\
    where the probs are encoded by the policy network.
    -------------------
    Args:

    policy: class object 
    state : ndarray provided by the env.step

    Returns:

    action : int

    ------------------
    """
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    actions_distribution = Categorical(probs)
    action = actions_distribution.sample()
    policy.saved_log_probs.append(actions_distribution.log_prob(action))
    return action.item()

def calculate_policy_loss(policy):
    """
    Calculates policy loss. \
    If our method is REINFORCE then this is computed as the negative product of the log probs times the returns.\
    If our method is Fd then it is simply computed as the total rewards. 
    -----------------
    Args:

    policy : class object

    Returns:

    loss: torch.tensor

    -------------------
    """
    eps = np.finfo(np.float32).eps.item()
    R = 0
    policy_loss_list = [] 
    future_returns = []
    for r in policy.rewards[::-1]: # reverse buffer r
        R = r + args.gamma * R # G_t = r_t + gamma*G_{t+1}
        future_returns.insert(0, R) # insert at the beginning
    future_returns = torch.tensor(future_returns)
    future_returns = (future_returns - future_returns.float().mean()) / (future_returns.float().std() + eps) # normalize returns
    if args.method == 'reinforce' or args.method == 'fd_with_reinforce':
        for log_prob, Gt in zip(policy.saved_log_probs, future_returns): # Use trajectory to estimate the policy loss
            policy_loss_list.append(-log_prob * Gt) # policy loss is the negative log probability of the action times the discounted return
    elif args.method == 'fd':
        for Gt in future_returns:
            policy_loss_list.append(Gt)
    else:
        raise ValueError('This method is not valid. Please try a different one.')
    return torch.cat(policy_loss_list).sum() if args.method == 'reinforce' else torch.tensor(policy_loss_list).sum()# sum up gradients

def deltas(epsilon, gamma, episode):
    """
    This function returns the perdubations delta sampled from a uniform distribution [-epsilon, epsilon]
    ----------------
    Args:

    env: environment
    gamma: scalar
    episode: scalar (number of current episode)

    Returns:

    delta: torch.tensor (pertubation)
    """
    delta = Uniform(-epsilon, epsilon).sample()
    gamma = gamma**episode
    delta = delta * gamma
    return torch.max(torch.tensor([delta, torch.finfo(torch.double).eps])).double()


def update_policy_reinforce(env, policy, optimizer):
    """
    Updates policy when using the REINFORCE method.
    """
    policy_loss = calculate_policy_loss(policy)
    optimizer.zero_grad() # clear gradients
    policy_loss.backward() # backpropagate
    optimizer.step() # update policy
    del policy.rewards[:] # clear rewards
    del policy.saved_log_probs[:] # clear log_probs


def update_policy_fd(env, policy, episode, lr, epsilon=0.1, no_of_pertubations=2, gamma=0.9):
    """
    Updates policy when using the fd method based on :https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.78.6163&rep=rep1&type=pdf
    """
    policy_loss = calculate_policy_loss(policy)
    policy_loss_gradient = torch.zeros_like(policy.linear1.weight.data)
    for dim1 in range(policy.linear1.weight.data.shape[0]):
        for dim2 in range(policy.linear1.weight.data.shape[1]):
            policy_losses = []
            pertubations = []
            for k in range(no_of_pertubations):
                #copy the policy
                pertubated_policy = Policy()
                #copy the environment
                env_copy = copy.deepcopy(env)
                sample_episode(env_copy, pertubated_policy)
                #sample perturbations uniformly from [-epsilon, epsilon]
                pertubation = deltas(epsilon, gamma, episode)
                pertubations.append(pertubation)
                pertubated_policy.linear1.weight.data[dim1][dim2] = policy.linear1.weight.data[dim1][dim2] + pertubation
                pert_policy_loss = calculate_policy_loss(pertubated_policy)
                policy_losses.append(pert_policy_loss-policy_loss)
            policy_loss_gradient[dim1][dim2] = torch.dot(torch.tensor(policy_losses).float(), torch.tensor(pertubations).float()).sum() / torch.sum(torch.tensor(pertubations)**2)
    policy.linear1.weight.data += - lr * policy_loss_gradient # gradient descent

    #clear rewards
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def sample_episode(env, policy, render=False):
    """
    Samples episode
    """
    state, ep_reward = env.reset(), 0
    for t in range(1, 10000): # Don't infinite loop while learning
        if args.env == 'CartPole-v0-custom':
            force = select_force(policy, state)
            state, reward, done, _ = env.step(force)
        else:
            action = select_action(policy, state)
            state, reward, done, _ = env.step(action)
        if render:
            env.render()
        policy.rewards.append(reward)
        ep_reward += reward
        if done:
            break
    return t, ep_reward

def main(seed, number_of_episodes=200):
    env = gym.make(args.env) if args.env != 'CartPole-v0-custom' else CartPolev0()
    env.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    policy = Policy()
    optimizer = optim.SGD(policy.parameters(), lr=args.lr) #only used for reinforce
    running_reward = 10
    episode_rewards = []
    for i_episode in range(number_of_episodes):
        state, ep_reward = env.reset(), 0
        #sample an episode
        last_episode_final_step, ep_reward = sample_episode(env, policy, render=args.render)
        episode_rewards.append(ep_reward)
        # running reward across episodes
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        # update policy
        if args.method == 'reinforce':
            update_policy_reinforce(env, policy, optimizer)
        elif args.method == 'fd' or args.method == 'fd_with_reinforce':
            update_policy_fd(env, policy, i_episode, args.lr)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
    return episode_rewards
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--method', type=str, required=True,
                        help='gradients calculation method (reinforce or fd or fd_with_reinforce)')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate')
    parser.add_argument('--env', default='CartPole-v0', required=True, 
                        help='name of the environment to run')
    parser.add_argument('--std', type=float, default=0.01, 
                        help='standard deviation used in create_force')
    parser.add_argument('--no_of_episodes', type=int, default=4000, required=True, metavar='N',
                        help='number of episodes to run expirements for')
    parser.add_argument('--no_of_pertubations', type=int, default=4,
                        help='number of pertubations')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='G',
                        help='discount factor (default: 0.9)')
    parser.add_argument('--epsilon', type=float, default=0.1, metavar='G',
                        help='factor used for uniformly distributed pertubations (default: 0.1)')
    parser.add_argument('--no_seeds', type=int, default=3, metavar='N',
                        help='number of seeds (default: 10)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    args = parser.parse_args()
    
    #Print the arguments
    print(args)

    #Run the experiment
    rewards = []
    #run main for multiple seeds
    for seed in range(args.no_seeds):
        rewards.append(main(seed, args.no_of_episodes))

    rewards = np.asarray(rewards)
    
    with open('rewards_pickle_{}_{}.pkl'.format(args.method, args.env), 'wb') as f:
       pickle.dump(rewards, f)