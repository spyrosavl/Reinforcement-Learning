import torch, pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

with open('rewards_pickle.pkl', 'rb') as f:
   rewards = pickle.load(f)

no_of_episodes = 4000
means = []; stds = []
for i in range(rewards.shape[1]):
    means.append(np.asarray(rewards[:,i]).mean())
    stds.append(np.asarray(rewards[:,i]).std())

#plot the rewards
plt.plot(torch.arange(1, no_of_episodes+1), torch.tensor(rewards).float().mean(dim=0))
plt.title('Average rewards')
plt.show()

X = np.arange(rewards.shape[1])
y_upper = np.array([means[i] + np.sqrt(stds[i]) for i in range(len(means))])
y_lower = np.array([means[i] - np.sqrt(stds[i]) for i in range(len(means))])

plt.figure(figsize=(20,10))


def downsample(factor,values):
    buffer_ = deque([],maxlen=factor)
    downsampled_values = []
    for i,value in enumerate(values):
        buffer_.appendleft(value)
        if (i-1)%factor==0:
            #Take max value out of buffer
            # or you can take higher value if their difference is too big, otherwise just average
            downsampled_values.append(max(buffer_))
    return np.array(downsampled_values)


plt.plot(downsample(50, X), downsample(50, means), '-r', label='Reward mean', c='orange', alpha=0.8)
# plt.plot(X, means, '-r', label='Reward mean', c='orange', alpha=0.8)
plt.fill_between(downsample(50, X), downsample(50, y_upper), downsample(50, y_lower), label='Reward std', facecolor='orange', alpha=0.3)

plt.xlabel("Episode")
# plt.xticks(range(len(X)), X)
plt.ylabel("Average reward")
plt.legend(loc='upper left')
plt.show()