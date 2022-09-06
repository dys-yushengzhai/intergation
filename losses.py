import torch
import torch.nn as nn
from utils_file.utils import *

def loss_normalized_cut(y_pred, graph):
    y = y_pred
    d = degree(graph.edge_index[0], num_nodes=y.size(0))
    gamma = y.t() @ d
    c = torch.sum(y[graph.edge_index[0], 0] * y[graph.edge_index[1], 1])
    return torch.sum(torch.div(c, gamma)) 

def loss_embedding(x,L):
    mse=nn.MSELoss()
    l=torch.tensor(0.)
    for i in range(x.shape[1]):
        l+=residual(x[:,i],L,mse)
    return l
