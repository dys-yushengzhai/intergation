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
from kmeans_pytorch import kmeans

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
            if config.data == 'all':
                update = update=torch.tensor(5).to(device)
                print_loss=5
                losses=[]
                for i in range(config.se_epoch):
                    loss = torch.tensor(0.).to(device)
                    j = 0
                    for d in config.loader:
                        d = d.to(device)
                        L = laplacian(d).to(device)
                        x = f(d)
                        loss += loss_fn(x,L,config.hyper_para_loss_embedding)/update
                        j+=1
                        if j%update.item()==0 or j==len(config.loader):
                            optimizer.zero_grad()
                            losses.append(loss.item())
                            loss.backward()
                            optimizer.step()
                            loss=torch.tensor(0.).to(device)
                    
                    print('Epoch:',i,'   Loss:',losses[-1])
                    # if i==config.se_epoch-1:
                    #     with open('x_features_epoch_'+str(config.se_epoch)+'.pickle','wb') as f:
                    #         pickle.dump(x,f)
            else:
                for i in range(config.se_epoch):
                    for d in config.loader:
                        d = d.to(device)
                        L = laplacian(d).to(device)
                        x = f(d)
                        loss = loss_fn(x,L,config.hyper_para_loss_embedding)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        print('Epoch:',i,'   Loss:',loss)
            print('End training')
            print('')

            # Save the model
            torch.save(f.state_dict(), config.se_train_savepath)
            print('Model saved')
            print('')
        
        if config.is_pe:
            dataset = []
            print('Start partitioning embedding module')
            f = ModelSpectral(config.se_params, device).to(device)
            f.load_state_dict(torch.load(config.se_train_savepath))
            f.eval()
            for p in f.parameters():
                p.requires_grad = False
            f.eval()
            dataset = []
            for d in config.loader:
                x = f(d)[:,1].reshape((d.num_nodes,1))
                x = (x-torch.mean(x))*\
                torch.sqrt(torch.tensor(d.num_nodes))
                d = Batch(x,edge_index=d.edge_index)
                dataset.append(d)
            config.loader = DataLoader(dataset,batch_size=1,shuffle=False,pin_memory=False)

            # for adj in config.dataset:
            #     adj = st.to_scipy(adj,layout='coo')
            #     row = adj.row
            #     col = adj.col
            #     rowcols = np.array([row,col])
            #     edges=torch.tensor(rowcols,dtype=torch.long)
            #     nodes = torch.randn(adj.shape[0],2)

            #     graph = Batch(x=nodes,edge_index = edges).to(device)
            #     graph.x = f(graph)[:,1].reshape((graph.num_nodes,1))
            #     graph.x = (graph.x-torch.mean(graph.x))*torch.sqrt(torch.tensor(graph.num_nodes))
            #     dataset.append(graph)
            # loader = DataLoader(dataset,batch_size=1,shuffle=False)
                # d = d.to(device)
                # L = laplacian(d).to(device)
                # x = f(d)
                # x = x[:,1:]
                # x = torch.randn(118758,1)
                # print(x)
                # x,cluster_centers = kmeans(
                #     X = x,num_clusters=2,distance='euclidean',tol=0.1,device='cpu')
                # x = torch.unsqueeze(x.float(), dim=1)
                # print(x)
            #     x = (x-torch.mean(x)) * \
            #         torch.sqrt(torch.tensor(x.shape[0]))
            #     dataset.append(Data(x=x,edge_index=d.edge_index))
            # loader = DataLoader(dataset,batch_size=1,shuffle=False)
        
            f_lap = ModelPartitioning(config.pe_params)
            f_lap.train()
            loss_fn = loss_normalized_cut
            optimizer = torch.optim.Adam(f_lap.parameters(),**config.pe_opt)
            max_cut = 1000000.
            if config.data=='all':
                update = torch.tensor(5).to(device)
                losses = []
                print_loss = 10
                for i in range(config.pe_epoch):
                    loss = torch.tensor(0.).to(device)
                    j = 0
                    for d in config.loader:
                        d = d.to(device)
                        data = f_lap(d)
                        loss += loss_fn(data,d,config.hyper_para_loss_normalized_cut)/update
                        j+=1
                        if j%update.item()==0 or j==len(config.loader):
                            optimizer.zero_grad()
                            losses.append(loss.item())
                            loss.backward()
                            optimizer.step()
                            loss=torch.tensor(0.).to(device)

                    if i%print_loss==0:
                        print('Epoch:',i,'   Loss:',losses[-1])
                            # save_model
                torch.save(f_lap.state_dict(), config.pe_train_savepath)
                print('Model saved')
                print('')
            else:
                for i in range(config.pe_epoch):
                    for d in config.loader:
                        d = d.to(device)
                        data = f_lap(d)
                        _,_,_,cuts,t,ias,ibs,ia,ib= best_part(data,d,2)
                        writer.add_scalars(config.plot_path,
                        {'cuts':cuts,'metis':config.baseline},i)

                        print('cut: ',cuts)
                        print('ia: ',len(ia))
                        print('ib: ',len(ib))
                        loss = loss_fn(data,d,config.hyper_para_loss_normalized_cut)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        print('Epoch:',i,'   Loss:',loss)
                        if(int(cuts)!=0 and cuts<max_cut and len(ib)!=0 and ((len(ia))/(len(ib))<1.15) and (len(ia))/(len(ib))>0.85 ):
                            max_cut = cuts
                            torch.save(f_lap.state_dict(), config.pe_train_savepath)
                            print('Model saved')
                            print('')
            


def testing(config):
    if config.mode=='test' and config.model == "spectral for graph embedding":
        device = config.device
        f = ModelSpectral(config.se_params, device).to(device)
        f.load_state_dict(torch.load(config.se_train_savepath))
        f.eval()
        for p in f.parameters():
            p.requires_grad = False
        f.eval()
        dataset = []
        for d in config.loader:
            x = f(d)[:,1].reshape((d.num_nodes,1))
            x = (x-torch.mean(x))*torch.sqrt(torch.tensor(d.num_nodes))
            d = Batch(x,edge_index=d.edge_index)
            dataset.append(d)
        config.loader = DataLoader(dataset,batch_size=1,shuffle=False,pin_memory=False)

        f_lap = ModelPartitioning(config.pe_params).to(device)
        f_lap.load_state_dict(torch.load(config.pe_train_savepath))
        f_lap.eval()
        for p in f_lap.parameters():
            p.requires_grad = False
        f_lap.train()
        for d in config.loader:
            d = d.to(device)
            data = f_lap(d)
            _,_,_,cuts,t,ias,ibs,ia,ib= best_part(data,d,2)
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

    # mode
    mode = 'test'

    # data: all or only one
    # data = 'all'
    # NotreDame_actors 卡主
    data_train = 'all'
    data_test = 'offshore'
    data = data_train if mode=='train' else data_test

    # the constraint of the nodes of graph
    n_max = 5000000
    n_min = 100

    # model
    model = 'spectral for graph embedding'

    # whether to initiate picture
    init_plot = False

    # 删除文件，用断点删除 , mode = 'all in folder','file','folder'
    if init_plot:
        remove_file_or_folder('log/'+data,mode='all in folder')

    # config
    config  = config_gap(data=data,batch_size=1,mode=mode,n_max=n_max,n_min=n_min)
    config.data = data
    config.is_plot = True
    config.model = model
    config.plot_path = 'log/'+config.data+'/'

    # print the message of dataset
    print_message(data,mode, config)
    
    # spectral embedding optimizer == se_opt(dict)(lr,weight_decay)
    config.se_opt = {'lr':0.001,'weight_decay':5e-6}
    # partitioning embdding optimizer == pm_opt(dict)(lr,weight_decay)
    config.pe_opt = {'lr':0.001,'weight_decay':5e-6}
    # whether to run spectral embedding
    config.is_se = False
    # whether to run partitiong embedding 
    config.is_pe = False
    config.hyper_para_loss_embedding = 1
    config.hyper_para_loss_normalized_cut = 0
    config.se_params = {'l':32,'pre':2,'post':2,'coarsening_threshold':2,'activation':'tanh','lins':[16,32,32,16,16]}
    config.pe_params = {'l':32,'pre':4,'post':4,'coarsening_threshold':2,'activation':'tanh','lins':[16,16,16,16,16]}
    config.se_epoch = 120 # 120 80(0.001)
    config.pe_epoch = 150 # 150 # 100(0.0005)
    config.se_train_savepath = 'spectral_weights/spectral_weights_'+data_train+'.pt'
    config.pe_train_savepath  = 'partitioning_weights/partitioning_weights_'+data_train+'.pt'
    config.se_test_savepath = 'spectral_weights/spectral_weights_'+data_test+'.pt'
    config.pe_test_savepath = 'partitioning_weights/partitioning_weights_'+data_test+'.pt'


    writer = SummaryWriter(config.plot_path)
    train(config)
    # config.mode='test'
    testing(config)
    writer.close()

# 'imagesensor'
# metis:8230
# 50 0.00100930733606219291687011718750 14145
# 51 0.00118701963219791650772094726562 15200
# 60 0.00139058567583560943603515625000 16823
# 80 0.00110348907765001058578491210938 8196
# 100 0.00130757794249802827835083007812 8810.0
# 150 0.00154331908561289310455322265625 11242.0
# 108 0.00088301947107538580894470214844 11737.0
# 200 0.00127180945128202438354492187500 9620.0

# 'power9'
# metis:5871
# 50 0.00071260012919083237648010253906 5652.0
# 60 0.00074137421324849128723144531250 8934.0

# 'ss1'
# metis:1398
# 50 0.00187207432463765144348144531250 1661.0
# 51 0.00204618135467171669006347656250 4316.0
# 60 0.00226833322085440158843994140625 3961.0
# 40 0.00141398562118411064147949218750 5126.0
# 49 0.00081431370927020907402038574219 7923.0

# 'radiation'
# metis:11371
# 50 0.00082438124809414148330688476562 12711.0
# 55 0.00142273271922022104263305664062 12230.0
# 60 0.00167087116278707981109619140625 12551.0
# 56 0.00068081705830991268157958984375 76747.0