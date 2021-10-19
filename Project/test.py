import torch
import numpy as np
from torch.distributions.uniform import Uniform
import torch.nn as nn 

epsilon = np.finfo(np.float32).eps.item()
no_of_pertubations = 2
eps = 0.1
perdubations = (torch.rand(no_of_pertubations) * 2 * epsilon - epsilon).tolist()
pertubations = Uniform(-eps, eps).sample((2,1))
print(perdubations, pertubations)