import os
from utils_file.utils import *
from scipy.io import mmread
import torch_sparse
from torch_sparse import SparseTensor as st
import networkx as nx
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import random
import pickle
import numpy


def mixed_dataset(data='all',binary_weight=True,batch_size=1,n_max=5000,n_min=100):
    dataset_list = []
    dataset = []
    # if file_exist(pkl_path)
    #     :
    # else:
    if data=='all':
        pickle_path = 'pickle_file/pickle_all/'
        data_path = 'data/data_all/'
        if os.path.isfile(pickle_path+data+'.pickle'):
            with open (pickle_path+data+'.pickle','rb') as f:
                pickle_data = pickle.load(f)
                return pickle_data
        else:         
            for m in os.listdir(os.path.expanduser(data_path)):
                adj = mmread(os.path.expanduser(data_path+str(m)))
                if(isinstance(adj,numpy.ndarray)):
                    continue
                if adj.shape[0]<n_max and adj.shape[0]>n_min:
                    print(m)
                    adj = torch_sparse.remove_diag(st.from_scipy(adj)).to_symmetric()
                    adj,is_connected = to_continuous_natural_number_and_is_connected(adj)
                    if not is_connected:
                        if binary_weight:
                            adj = st.set_value(adj, torch.ones_like(adj.storage._value),layout='coo')
                            for i in range(3):
                                dataset_list.append(adj)

                        else:
                            dataset_list.append(adj)
                    
                    else:
                        raise Exception('have isolated nodes')

            for adj in dataset_list:
                adj = st.to_scipy(adj,layout='coo')
                row = adj.row
                col = adj.col
                rowcols = np.array([row,col])
                edges = torch.tensor(rowcols,dtype=torch.long)
                nodes = torch.ones(adj.shape[0],2)
                dataset.append(Data(x=nodes, edge_index=edges))
            loader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
                
            return loader
    else:
        pickle_path = 'pickle_file/pickle/'
        data_path = 'data/suitesparse/'
        if os.path.isfile(pickle_path+data+'.pickle'):
            with open(pickle_path+data+'.pickle','rb') as f:
                pickle_data = pickle.load(f)
                return pickle_data
        else:
            adj = mmread(os.path.expanduser('data/suitesparse'+'/'+data+'.mtx'))
            adj = torch_sparse.remove_diag(st.from_scipy(adj)).to_symmetric()
            adj,is_connected = to_continuous_natural_number_and_is_connected(adj)
            if not is_connected:
                if binary_weight:
                    adj = st.set_value(adj, torch.ones_like(adj.storage._value),layout='coo')
                    dataset_list.append(adj)

                else:
                    dataset_list.append(adj)
            
            else:
                raise Exception('have isolated nodes')

            for adj in dataset_list:
                adj = st.to_scipy(adj,layout='coo')
                row = adj.row
                col = adj.col
                rowcols = np.array([row,col])
                edges = torch.tensor(rowcols,dtype=torch.long)
                nodes = torch.ones(adj.shape[0],2)
                dataset.append(Data(x=nodes, edge_index=edges))
            loader = DataLoader(dataset,batch_size=batch_size,shuffle=False)

            with open(pickle_path+data+'.pickle','wb') as f:
                pickle.dump(loader,f)
            
            return loader
