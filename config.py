import torch
from utils_file.dataset_utils import mixed_dataset
from utils_file.utils import *
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

class config_gap:
    def __init__(self,data='ss1',batch_size=1,mode = 'train',is_plot=False):
        self.model = "spectral for graph embedding"
        self.loader,self.dataset = mixed_dataset(data,batch_size=batch_size)
        self.data = data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.is_plot = is_plot
        self.plot_path = None
        self.baseline = 0
        self.balance = 0

        if self.model=="spectral for graph embedding" and self.mode=='train':
            # spectral embedding optimizer == se_opt
            self.se_opt = {'lr':0.001,'weight_decay':5e-5}
            # partitioning embedding optimizer == pm_opt
            self.pe_opt = {'lr':0.01,'weight_decay':5e-6}
            self.is_se = True
            self.is_pe = True
            self.se_params = {'l':32,'pre':4,'post':4,'coarsening_threshold':2,'activation':'tanh','lins':[16,32,32,16,16]}
            self.pe_params = {'l':32,'pre':4,'post':4,'coarsening_threshold':2,'activation':'tanh','lins':[16,16,16,16,16]}
            self.se_epoch = 200
            self.pe_epoch = 200
            self.se_savepath = 'spectral_weights/spectral_weights_ss1.pt'
            self.pe_savepath = 'partitioning_weights/partitioning_weights_ss1.pt'
            self.hyper_para_loss = 0
        elif self.model=="spectral graph embedding" and self.mode=='test':
            pass  


# device = 'cpu'
# A = input_matrix()
# print(A.toarray())
# row = A.row
# col = A.col
# rowcols = np.array([row,col])
# edges = torch.tensor(rowcols,dtype=torch.long)
# nodes = torch.randn(A.shape[0],2)
# data = Data(x=nodes,edge_index=edges)
# print(data)
# dataset = []
# dataset.append(data)
# loader = DataLoader(dataset,batch_size=1,shuffle=True)
# print(loader)
# for d in loader:
#     print(laplacian(d))
    
    
# print(g.edge_index)
# print(laplacian(A))
# config = config_gap()
# print(config.dataset)
# for d in config.loader:
#     print(d)