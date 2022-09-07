from __future__ import division
import torch
from utils_file.dataset_utils import mixed_dataset
from utils_file.utils import *
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pickle

import torch
import torch.nn as nn
from layers import *
from torch_geometric.nn import  avg_pool, graclus
from torch_geometric.data import Batch
from layers import SAGEConv
import pickle
import random

# data = 'imagesensor'

# loader = mixed_dataset(data)

# print(loader)
torch.set_printoptions(precision=32)
with open('x_no_pickle.pickle','rb') as f:
    x_no_pickle = pickle.load(f)

with open('x_pickle.pickle','rb') as f:
    x_pickle = pickle.load(f)

print(x_no_pickle[0][0])

print(x_pickle[0][0])

# tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],[1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]])
# tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],[1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]])


print(x_no_pickle[1][0])
print(x_pickle[1][0])

torch.manual_seed(176364)
np.random.seed(453658)
random.seed(41884)
torch.cuda.manual_seed(9597121)

for i in range(10):
    print(torch.randn(1,1))