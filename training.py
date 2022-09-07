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
import pickle

torch.set_printoptions(precision=32)

def train(config):
    device = config.device
    # spectral for graph embedding
    if config.mode=='train' and config.model == "spectral for graph embedding":
        if config.is_se:
            f = ModelSpectral(config.se_params,device).to(device)
            f.train()
            print('Number of parameters:',sum(p.numel() for p in f.parameters()))
            print('')
            optimizer = torch.optim.Adam(f.parameters(),**config.se_opt)
            loss_fn = loss_embedding
            print('Start spectral embedding module')
            print(' ')
            for i in range(config.se_epoch):
                for d in config.loader:
                    d = d.to(device)
                    L = laplacian(d).to(device)
                    x = f(d)
                    loss = loss_fn(x,L)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print('Epoch:',i,'   Loss:',loss)
            print('End training')
            print('')

            # Save the model
            torch.save(f.state_dict(), config.se_savepath)
            print('Model saved')
            print('')
        
        if config.is_pe:
            print('Start partitioning embedding module')
            f = ModelSpectral(config.se_params, device).to(device)
            f.load_state_dict(torch.load(config.se_savepath))
            f.eval()
            for p in f.parameters():
                p.requires_grad = False
            f.eval()
            dataset = []
            for d in config.loader:
                d = d.to(device)
                L = laplacian(d).to(device)
                x = f(d)
                x = x[:,1:]
                x = (x-torch.mean(x)) * \
                    torch.sqrt(torch.tensor(x.shape[0]))
                dataset.append(Data(x=x,edge_index=d.edge_index))
            loader = DataLoader(dataset,batch_size=1,shuffle=False)

            f_lap = ModelPartitioning(config.pe_params)
            f_lap.train()
            loss_fn = loss_normalized_cut
            optimizer = torch.optim.Adam(f_lap.parameters(),**config.pe_opt)
            max_cut = 1000000.
            for i in range(config.pe_epoch):
                for d in loader:
                    d = d.to(device)
                    data = f_lap(d)
                    _,_,_,cuts,t,ia,ib,ic,id = best_part(data,d,2)
                    writer.add_scalars(config.plot_path,
                    {'cuts':cuts,'metis':config.baseline},i)

                    print('cut: ',cuts)
                    print('ia: ',len(ia))
                    print('ib: ',len(ib))
                    loss = loss_fn(data,d)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print('Epoch:',i,'   Loss:',loss)
                    if(int(cuts)!=0 and cuts<max_cut and len(ib)!=0 and ((len(ia))/(len(ib))<1.15) and (len(ia))/(len(ib))>0.85 ):
                        max_cut = cuts
                        torch.save(f_lap.state_dict(), config.pe_savepath)
                        print('Model saved')
                        print('')
            # torch.save(f_lap.state_dict(), config.pe_savepath)
            # print('Model saved')
            # print('')

def testing(config):
    if config.mode=='test' and config.model == "spectral for graph embedding":
        device = config.device
        f = ModelSpectral(config.se_params, device).to(device)
        f.load_state_dict(torch.load(config.se_savepath))
        f.eval()
        for p in f.parameters():
            p.requires_grad = False
        f.eval()
        dataset = []
        for d in config.loader:
            d = d.to(device)
            L = laplacian(d).to(device)
            x = f(d)
            x = x[:,1:]
            x = (x-torch.mean(x)) * \
                torch.sqrt(torch.tensor(x.shape[0]))
            dataset.append(Data(x=x,edge_index=d.edge_index))
        loader = DataLoader(dataset,batch_size=1,shuffle=False)
        f_lap = ModelPartitioning(config.pe_params).to(device)
        f_lap.load_state_dict(torch.load(config.pe_savepath))
        f_lap.eval()
        for p in f_lap.parameters():
            p.requires_grad = False
        f_lap.train()
        for d in loader:
            d = d.to(device)
            data = f_lap(d)
            _,_,_,cuts,t,ia,ib,ic,id = best_part(data,d,2)
            print('cut: ',cuts)
            print('ia: ',max(len(ia),len(ib)))
            print('ib: ',min(len(ia),len(ib)))
            print('balance: ',max(len(ia),len(ib))/min(len(ia),len(ib)))
        


if __name__ == '__main__':
    # Seeds
    # torch.manual_seed(176364)
    # np.random.seed(453658)
    # random.seed(41884)
    # torch.cuda.manual_seed(9597121)

    torch.manual_seed(1763)
    np.random.seed(453658)
    random.seed(41884)
    torch.cuda.manual_seed(9597121)

    # data: all or only one
    # data = 'all'
    data = 'imagesensor'
    init_plot = False

    # 删除文件，用断点删除 , mode = 'all in folder','file','folder'
    if init_plot:
        remove_file_or_folder('log/'+data,mode='all in folder')

    # config
    config  = config_gap(data=data,batch_size=1,mode='train')
    config.data = data
    config.is_plot = True
    config.plot_path = 'log/'+config.data+'/'
    config.baseline,config.balance = pymetis_baseline(config.dataset[0])
    print('data: ',config.dataset[0])
    print('')
    print('The number of nodes: ',config.dataset[0].storage._sparse_sizes[0])
    print('')
    print('The number of edges: ',int(config.dataset[0].storage._row.shape[0]/2))
    print('')
    print('metis cuts: ',config.baseline)
    print('metis balance: ',config.balance)
    config.model = 'spectral for graph embedding'
    # spectral embedding optimizer == se_opt(dict)(lr,weight_decay)
    config.se_opt = {'lr':0.001,'weight_decay':5e-6}
    # partitioning embdding optimizer == pm_opt(dict)(lr,weight_decay)
    config.pe_opt = {'lr':0.001,'weight_decay':5e-6}
    # whether to run spectral embedding
    config.is_se = False
    # whether to run partitiong embedding 
    config.is_pe = False
    config.se_params = {'l':32,'pre':2,'post':2,'coarsening_threshold':2,'activation':'tanh','lins':[16,32,32,16,16]}
    config.pe_params = {'l':32,'pre':4,'post':4,'coarsening_threshold':2,'activation':'tanh','lins':[16,16,16,16,16]}
    config.se_epoch = 100
    config.pe_epoch = 60
    config.se_savepath = 'spectral_weights/spectral_weights_'+config.data+'.pt'
    config.pe_savepath  = 'partitioning_weights/partitioning_weights_'+config.data+'.pt'
    print('Number of SuiteSparse graphs:',len(config.dataset))
    print('')

    writer = SummaryWriter(config.plot_path)
    train(config)
    config.mode='test'
    testing(config)
    writer.close()
    


     