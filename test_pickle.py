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
from kmeans_pytorch import kmeans


# Seeds
# torch.manual_seed(176364)
# np.random.seed(453658)
# random.seed(41884)
# torch.cuda.manual_seed(9597121)
np.random.seed(123)

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

# remove_file_or_folder('test_os/test_folder',mode='all in folder')
x = 0
with open('x_features_epoch_50.pickle','rb') as f:
    data = pickle.load(f)
    x = data[:,1:]
    x = (x-torch.mean(x)) * \
    torch.sqrt(torch.tensor(x.shape[0]))
print(x)
# cluster_ids_x,cluster_centers = kmeans(
#     X = x,num_clusters=2,distance='euclidean',device='cpu'
# )



# data_size,dims,num_clusters = 1000,2,3
# x = np.random.randn(data_size,dims)/6
# x = torch.from_numpy(x)
# cluster_ids_x,cluster_centers = kmeans(
#     X=x,num_clusters=num_clusters,distance='euclidean',device='cpu'
# )



# plt.figure(figsize=(4, 3), dpi=160)
# plt.scatter(x[:, 0], x[:, 1], c=cluster_ids_x, cmap='cool')
# plt.scatter(
#     cluster_centers[:, 0], cluster_centers[:, 1],
#     c='white',
#     alpha=0.6,
#     edgecolors='black',
#     linewidths=2
# )
# plt.axis([-1, 1, -1, 1])
# plt.tight_layout()
# plt.show()
