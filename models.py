import torch
import torch.nn as nn
from layers import *
from torch_geometric.nn import  avg_pool, graclus
from torch_geometric.data import Batch
from layers import SAGEConv
import pickle


# Neural network for the embedding module
class ModelSpectral(torch.nn.Module):
    def __init__(self,se_params,device):
        super(ModelSpectral,self).__init__()
        self.l = se_params.get('l')
        self.pre = se_params.get('pre')
        self.post = se_params.get('post')
        self.coarsening_threshold = se_params.get('coarsening_threshold')
        self.activation = getattr(torch, se_params.get('activation'))
        self.lins = se_params.get('lins')

        self.conv_post = nn.ModuleList(
            [SAGEConv(self.l, self.l) for i in range(self.post)]
        )
        self.conv_coarse = SAGEConv(2,self.l)
        self.lins1=nn.Linear(self.l,self.lins[0])
        self.lins2=nn.Linear(self.lins[0],self.lins[1]) 
        self.lins3=nn.Linear(self.lins[1],self.lins[2]) 
        self.final=nn.Linear(self.lins[2],2)
        self.device = device

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        unpool_info = []
        x_info=[]
        cluster_info=[]
        edge_info=[]
        while x.size()[0] > self.coarsening_threshold:
            cluster = graclus(edge_index,num_nodes=x.shape[0])
            cluster_info.append(cluster)
            edge_info.append(edge_index)
            gc = avg_pool(cluster, Batch(batch=batch, x=x, edge_index=edge_index,shuffle=False))
            x, edge_index, batch = gc.x, gc.edge_index, gc.batch
        # coarse iterations
        x=torch.eye(2).to(self.device)
        x=self.conv_coarse(x,edge_index)
        x=self.activation(x)
        while edge_info:
            # un-pooling / interpolation / prolongation / refinement
            edge_index = edge_info.pop()
            cluster = cluster_info.pop()
            output, inverse = torch.unique(cluster, return_inverse=True)
            x = x[inverse]
            # post-smoothing
            for i in range(self.post):
                x = self.activation(self.conv_post[i](x, edge_index))
        x=self.lins1(x)
        x=self.activation(x)
        x=self.lins2(x)
        x=self.activation(x)
        x=self.lins3(x)
        x=self.activation(x)
        x=self.final(x)
        x,_=torch.linalg.qr(x,mode='reduced')
        return x

# Neural network for the partitioning module
class ModelPartitioning(torch.nn.Module):
    def __init__(self,pe_params):
        super(ModelPartitioning,self).__init__()

        self.l = pe_params.get('l')
        self.pre = pe_params.get('pre')
        self.post = pe_params.get('post')
        self.coarsening_threshold = pe_params.get('coarsening_threshold')
        self.activation = getattr(torch, pe_params.get('activation'))
        self.lins = pe_params.get('lins')

        self.conv_first = SAGEConv(1, self.l)
        self.conv_pre = nn.ModuleList(
            [SAGEConv(self.l, self.l) for i in range(self.pre)]
        )
        self.conv_post = nn.ModuleList(
            [SAGEConv(self.l, self.l) for i in range(self.post)]
        )
        self.conv_coarse = SAGEConv(self.l,self.l)
        
        self.lins1=nn.Linear(self.l,self.lins[0])
        self.lins2=nn.Linear(self.lins[0],self.lins[1]) 
        self.lins3=nn.Linear(self.lins[1],self.lins[2]) 
        self.final=nn.Linear(self.lins[4],2)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = self.activation(self.conv_first(x, edge_index))
        unpool_info = []
        x_info=[]
        cluster_info=[]
        edge_info=[]
        batches=[]
        while x.size()[0] > self.coarsening_threshold:
            # pre-smoothing
            for i in range(self.pre):
                x = self.activation(self.conv_pre[i](x, edge_index))
            # pooling / coarsening / restriction
            x_info.append(x)
            batches.append(batch)
            cluster = graclus(edge_index,num_nodes=x.shape[0])
            cluster_info.append(cluster)
            edge_info.append(edge_index)
            gc = avg_pool(cluster, Batch(batch=batch, x=x, edge_index=edge_index))
            x, edge_index, batch = gc.x, gc.edge_index, gc.batch
        # coarse iterations
        x = self.activation(self.conv_coarse(x,edge_index))
        while edge_info:
            # un-pooling / interpolation / prolongation / refinement
            edge_index = edge_info.pop()
            output, inverse = torch.unique(cluster_info.pop(), return_inverse=True)
            x = (x[inverse] + x_info.pop())/2
            # post-smoothing
            for i in range(self.post):
                x = self.activation(self.conv_post[i](x, edge_index))
        x=self.lins1(x)
        x=self.activation(x)
        x=self.lins2(x)
        x=self.activation(x)
        x=self.lins3(x)
        x=self.activation(x)
        x=self.final(x)
        x=torch.softmax(x,dim=1)
        return x

