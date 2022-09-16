from __future__ import division
import torch
from utils_file.dataset_utils import mixed_dataset
from utils_file.utils import *
from utils_file.dataset_utils import *
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
from torch_geometric.utils import degree

import scipy
from layers import *
from config import config_gap
from losses import *
# # data = 'imagesensor'

# # loader = mixed_dataset(data)

# # print(loader)
# torch.set_printoptions(precision=32)
# with open('x_no_pickle.pickle','rb') as f:
#     x_no_pickle = pickle.load(f)

# with open('x_pickle.pickle','rb') as f:
#     x_pickle = pickle.load(f)

# print(x_no_pickle[0][0])

# print(x_pickle[0][0])

# # tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],[1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]])
# # tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],[1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]])


# print(x_no_pickle[1][0])
# print(x_pickle[1][0])

# torch.manual_seed(176364)
# np.random.seed(453658)
# random.seed(41884)
# torch.cuda.manual_seed(9597121)

# for i in range(10):
#     print(torch.randn(1,1))

# y = torch.tensor([[1,0],[1,0],[1,0],[1,0],[1,0]])
# node_num = y.shape[0]
# partition_num = y.shape[1]
# y = torch.sum(y,axis=0)

# print(node_num/partition_num)
# y = y-node_num/partition_num
# print(y)
# print(torch.sum(y.pow(2),axis=0))
# # print(y.size(0)/y.size(1))
# # print(y)

# node_num = y.shape[0]
# print(node_num)
# partition_num = y.shape[1]

# y = torch.sum((torch.sum(y,axis=0)-node_num/partition_num).pow(2),axis=0)/(torch.tensor(node_num,dtype=torch.float32).pow(2)/2.)
# print(y)

# A = input_matrix()
# row = torch.from_numpy(A.row).long()
# col = torch.from_numpy(A.col).long()
# data = torch.from_numpy(A.data)
# edge_index = torch.vstack((row,col))
# print(edge_index)
# matrix = st.from_edge_index(edge_index=edge_index,edge_attr=data)
# print(matrix.sparse_size(dim=0))
# d = degree(edge_index[0],num_nodes=matrix.sparse_size(dim=0))
# d = torch.tensor([2.,2.,4.,3.,3.,2.,2.])
# print((d==0.).any())



class GNet(nn.modules):
    def __init__(self,in_dim,n_classes,args):
        super(GNet,self).__init__()
        self.n_act = getattr(nn, args.act_n)()
        self.c_act = getattr(nn, args.act_c)()


data = 'imagesensor'

init_plot = False

config  = config_gap(data=data,batch_size=1,mode='train')
print(config.dataset)
for d in config.loader:
    print(d.edge_index)


