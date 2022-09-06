import pickle
import math

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
import scipy
from layers import *
from config import config_gap
from losses import *
import random
import torch
import losses
from models import *
from utils_file.utils import *
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter
import os
from utils_file.dataset_utils import mixed_dataset


# Seeds
torch.manual_seed(176364)
np.random.seed(453658)
random.seed(41884)
torch.cuda.manual_seed(9597121)

# config  = config_gap(data_path='data/suitesparse',batch_size=1,mode='train')

# print(config.dataset)

# pickle_path = 'ss1.pickle'
# # with open(pickle_path,'wb') as f:
# #     pickle.dump(config.loader, f,pickle.HIGHEST_PROTOCOL)

# with open(pickle_path,'rb') as f:
#     pickle_data = pickle.load(f)
#     print(pickle_data)


# for d in pickle_data:
#     print(d)

# def to_pickle(config.loader):
# print(os.path.isfile('pickle_file/'+'ss1'+'.pickle'))

# print(os.path.expanduser('pickle_file/'+'ss1'+'.pickle'))

# loader,dataset = mixed_dataset(data='ss1')

# with open('wode.pickle','wb') as f:
#     pickle.dump([1,2,3,4,6],f)

# with open('wode.pickle','rb') as f:
#     data = pickle.load(f)

# print(data)

remove_file_or_folder('test_os/test_folder',mode='all in folder')