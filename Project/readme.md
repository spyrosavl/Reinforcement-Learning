# Running the experiments

In this experiment we are comparing two gradient policy methods, namely, REINFORCE and Finite differences. This comparison is made across 3 different environments. 

Below the reader can find the instructions of excecution. 

This code is also accessible on [Github](https://github.com/spyrosavl/Reinforcement-Learning/tree/master/Project)

1. Make sure you are working in an environment containing the right packages:

`$ conda env create --name envname --file=env.yml`

**Note** The *env.yml* file is provided in the repository.

2. To run the experiments simply run:

`$ python3 fd_vs_reinforce.py --method method --env environment --no_of_episodes no_of_episodes`

In our code we will refer to REINFORCE as reinforce, Finite differences as fd and Finite differences with REINFORCE loss as fd_with_reinforce. Moreover, we experiment in the following environments:

- CartPole-v0-custom (for more details refer to our report [here](https://www.overleaf.com/read/tppwsbmjbynt), or, check *environment.py*)
- CartPole-v0
- LunarLander-v2

For example, if one would like to run our experiments for CartPole-v0 he/she should run, without significance on the order, the following:

`$ python3 fd_vs_reinforce.py --method reinforce --env CartPole-v0 --no_of_episodes 4000`

and 

`$ python3 fd_vs_reinforce.py --method fd --env CartPole-v0 --no_of_episodes 4000`

Each experiment is excecuted across 3 random seeds and at the end of **each** excecution a *reward_pickle_{method}_{env}.pkl* file is created. The reader can make use of this file by running the script in *plot_creation.py* to recreate the plots in our report.

The hyperparameters of the experiments are fixed and chosen empirically. If the reader is interested in changing any of the hyperparameters, they are registered at the end of *fd_vs_reinforce.py*


