import torch
import torch.nn as nn
from utils_file.utils import *

def loss_normalized_cut(y_pred, graph,Lambda=0):
    y = y_pred
    d = degree(graph.edge_index[0], num_nodes=y.size(0))
    gamma = y.t() @ d
    c = torch.sum(y[graph.edge_index[0], 0] * y[graph.edge_index[1], 1])
    node_num = y.shape[0]
    partition_num = y.shape[1]
    return torch.sum(torch.div(c, gamma)) + Lambda * torch.sum((torch.sum(y,axis=0)-node_num/partition_num).pow(2),axis=0)/(torch.tensor(node_num,dtype=torch.float32).pow(2)/2.)

def loss_embedding(x,L):
    mse=nn.MSELoss()
    l=torch.tensor(0.)
    for i in range(x.shape[1]):
        l+=residual(x[:,i],L,mse)
    return l
