import math
import numpy as np
import torch
import scipy.sparse as sp
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
# from models import *
from torch_sparse import SparseTensor as st
import torch_sparse
from tqdm.auto import tqdm
import pandas as pd
from torch_geometric.utils import to_networkx, degree, to_scipy_sparse_matrix, get_laplacian, remove_self_loops
import timeit
from collections import Counter
import pymetis
import os
import shutil
import gc
from torch_sparse import SparseTensor

def input_matrix():
    '''
    Returns a test sparse SciPy adjecency matrix
    '''

    N = 7
    data = np.ones(2 * 9)
    row = np.array([0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6])
    col = np.array([2, 3, 4, 6, 0, 4, 5, 6, 0, 4, 5, 1, 2, 3, 2, 3, 1, 2])

    A = sp.coo_matrix((data, (row, col)), shape=(N, N))

    return A

def symnormalize(A):
    """
    symmetrically normalise torch_sparse matrix

    arguments:
    M: torch sparse matrix

    returns:
    D^{-1/2} M D^{-1/2}
    where D is the diagonal node-degree matrix
    """

    
    d = torch_sparse.sum(A,1)
    dhi = torch.pow(d, -1/2).flatten()
    dhi[torch.isinf(dhi)] = 0
    dhi = dhi.numpy()
    DHI = sp.diags(dhi)
    
    return sparse_mx_to_torch_sparse_tensor((DHI.dot(st.to_scipy(A))).dot(DHI))

def normalize(A):
    """normalize A [0,1]

    Args:
        A (_type_): sparse matrix
    """
    


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch.sparse"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def torch_sparse_tensor_to_sparse_mx(torch_sparse_mx):
    """Convert torch.sparse to a scipy sparse

    Returns:
        scipy sparse
    """
    torch_sparse_mx_indices = torch_sparse_mx._indices()
    torch_sparse_mx_values = torch_sparse_mx._values()
    sparse_mx = sp.coo_matrix((torch_sparse_mx_values,(torch_sparse_mx_indices[0],torch_sparse_mx_indices[1])),(torch_sparse_mx.shape[0],torch_sparse_mx.shape[1]))
    return sparse_mx

def torch_sparse_add(torch_sparse_mx,num):
    return torch.sparse.FloatTensor(torch_sparse_mx._indices()
                                    ,torch_sparse_mx._values()+num,
                                    torch_sparse_mx.shape)

def abs_SparseTensor(torch_sparse_mx):
    """abs SparseTensor

    Returns:
        SparseTensor
    """
    return st.set_value(A, torch.abs(A.storage._value)) # SparseTensor

def test_partition(Y):
    """search the position of the maximum of probability

    Args:
        Y (_type_): 
            row : the number of nodes
            col : the number of class

    Returns:
            one dimension matrix
    """

    _, idx = torch.max(Y, 1)
    return idx

def proportion_partition(y):
    """The proportion of each partition

    Args:
        y probability matrix

    Returns:
        list[class1,class2,...]
    """
    classes_num = y.shape[1]
    nodes_num = y.shape[0]
    proportion_hashmap = {}

    # the position of maximum of probability
    idx = test_partition(y)
    idx = idx.tolist()

    for i in range(len(idx)):
        if not idx[i] in proportion_hashmap:
            proportion_hashmap[idx[i]]=1
        else:
            proportion_hashmap[idx[i]]=proportion_hashmap.get(idx[i])+1

    list_partition = [x/nodes_num for x in list(proportion_hashmap.values())]
    if len(list_partition)!=classes_num:
        for j in range(classes_num-len(list_partition)):
            list_partition.append(0)
            
    return sorted(list_partition,reverse = True)

def edge_cut(A,y):
    """Edge cut: the ratio of the cut to the total number of edges

    Args:
        A (torch_sparse):  For now, it's 01 adjacency matrix,no self loop
        y (tensor matrix): probability matrix

    Returns:
        cut/|E|
    """
    
    if isinstance(A, torch.Tensor):
        A = st.from_torch_sparse_coo_tensor(A)
    # n*1
    idx = torch.unsqueeze(test_partition(y), dim=1)
    # 0-->1 , 1-->0
    idx_not = torch.where((idx==0)|(idx==1),idx^1,idx).t()

    
    B = st.to_torch_sparse_coo_tensor(A)
    
    idx = torch.tensor(idx,dtype=torch.float32)

    idx_not = torch.tensor(idx_not,dtype=torch.float32)
    # B = torch.tensor(B,dtype=torch.float32)
    B = B.to(dtype=torch.float32)
    return (2*torch.mm(idx_not,torch.mm(B,idx))/A.storage._row.shape[0])

def from_txt_to_torch_sparse(data_path):
    """turn txt into torch_sparse

    Args:
        data_path (_type_): data path

    Returns:
        _type_: torch_sparse
    """

    data = pd.read_csv(data_path,sep=' ',header = None,names=['row','col','values'])
    data_tensor = torch.tensor(np.array(data))
    row,col,values = torch.split(data_tensor,(1,1,1),dim=1)
    indices = torch.vstack((row.t(),col.t()))
    values = torch.squeeze(values, dim=1).to(dtype=torch.float32)
    sparse_matrix = torch.sparse_coo_tensor(indices,values)
    

    return st.from_torch_sparse_coo_tensor(sparse_matrix)

def from_adjacency_matrix_to_adjacency_list(adjacency_matrix):
    """turn adjacency matrix into adjacency list(input of pymetis)
            adjacency_list = [np.array([4, 2, 1]),
                        np.array([0, 2, 3]),
                        np.array([4, 3, 1, 0]),
                        np.array([1, 2, 5, 6]),
                        np.array([0, 2, 5]),
                        np.array([4, 3, 6]),
                        np.array([5, 3])]
        n_cuts, membership = pymetis.part_graph(2, adjacency=adjacency_list)
        # n_cuts = 3
        # membership = [1, 1, 1, 0, 1, 0, 0]

        nodes_part_0 = np.argwhere(np.array(membership) == 0).ravel() # [3, 5, 6]
        nodes_part_1 = np.argwhere(np.array(membership) == 1).ravel() # [0, 1, 2, 4]

    Args:
        adjacency_matrix (_type_): torch_sparse

    Returns:
        _type_: adjacency_list(list)
    """
    dict_pymetis = {}
    row = adjacency_matrix.storage._row.tolist()
    col = adjacency_matrix.storage._col.tolist()

    for i in range(len(row)):
        if not row[i] in dict_pymetis:
            dict_list = [col[i]]
            dict_pymetis[row[i]]=dict_list
        else:
            dict_list = dict_pymetis[row[i]]
            dict_list.append(col[i])

    dict_list = []
    for value in dict_pymetis.values():
        dict_list.append(value)

    return dict_list

def to_continuous_natural_number_and_is_connected(adjacent_matrix):
    """1 3 5 5 5 to 0 1 2 2 2

    Args:
        adjacent_matrix (_type_): torch_sparse

    Returns:
        _type_: torch_sparse
    """

    row = adjacent_matrix.storage._row.tolist()
    col = adjacent_matrix.storage._col.tolist()

    dict_continuous_number = {}
    num = 0
    for i in range(len(row)):
        if not row[i] in dict_continuous_number:
            dict_continuous_number[row[i]] = num
            num = num+1
    
    row = torch.tensor([dict_continuous_number[row[i]] for i in range(len(row))])
    col = torch.tensor([dict_continuous_number[col[i]] for i in range(len(col))])
    is_connected = ((degree(row,row[-1]+1))==0.).any()

    return st.from_edge_index(edge_index=torch.vstack((row,col)),edge_attr=adjacent_matrix.storage._value),is_connected
# def edge_cut_test(A,y):

#     if isinstance(A, torch.Tensor):
#         A = st.from_torch_sparse_coo_tensor(A)
#     idx = test_partition(y)
#     idx = idx.tolist()
#     classes_hashmap = {}

#     for i in range(len(idx)):
#         if not idx[i] in classes_hashmap:
#             classes_hashmap[idx[i]] = [i]
#         else:
#             classes_hashmap[idx[i]].append(i)

#     # classes_hashmap = sorted(classes_hashmap.items(),key = lambda x : x[0])
    
#     row = A.storage._row
#     col = A.storage._col
#     cut = 0

#     for i in tqdm(range(row.shape[0])):
        
#         for j in range(y.shape[1]-1):
            
#             if row[i] in classes_hashmap[j]:

#                 for k in range(j+1,y.shape[1]):
                    
#                     if col[i] in classes_hashmap[k]:
                        
#                         cut = cut+1

#     return 2*cut/A.storage._row.shape[0]




# def Check(mat):
#     #iterative each row of COO
#     row_len
#     for r,c,value in range(nz):
#         row_len[r] = row_len[r]+1
#     for row in range(dim):
#         if row_len[row]==1:
#             print(row)


def laplacian(graph):
    lap=get_laplacian(graph.edge_index,num_nodes=graph.num_nodes)
    L=torch.sparse_coo_tensor(lap[0],lap[1])
    D=torch.sparse_coo_tensor(torch.stack((torch.arange(graph.num_nodes),torch.arange(graph.num_nodes)),dim=0),lap[1][-graph.num_nodes:])
    Dinv=torch.pow(D,-1)
    return torch.sparse.mm(Dinv,L)

# Computes the sum of the eigenvector residual and the eigenvalue related to the vector x
def residual(x,L,mse):
    return mse(L.matmul(x),rayleigh_quotient(x,L)*x)+rayleigh_quotient(x,L)

def rayleigh_quotient(x,L):
    return (torch.t(x).matmul(L.matmul(x))/(torch.t(x).matmul(x)))

def best_part(data, graph, n_times):
    ncuts = []
    vols = []
    preds = []
    cuts = []
    ias =[]
    ibs = [] 
    t0 = timeit.default_timer()
    graph_ev = data
    t1 = timeit.default_timer() - t0
    predictions = torch.argmax(graph_ev, dim=1)
    graph_pred = torch_from_preds(graph, predictions)

    # graph_pred.x = torch.tensor([0.,0.,0.,1.,1.,1.]) # only test,if real data comment out
    # vola total degree of partition a
    # volb total degree of partition complementary set a
    # cut edge between a and b
    
    nc_gap, vola, volb, cut,ia,ib= normalized_cut(graph_pred)
    ncuts.append(nc_gap)
    cuts.append(cut)
    vols.append((vola, volb))
    preds.append(predictions)
    ias.append(ia)
    ibs.append(ib)
    for i in range(n_times):
        t0_loop = timeit.default_timer()
        graph_ev = data
        t1_loop = timeit.default_timer() - t0_loop
        predictions = torch.argmax(graph_ev, dim=1)
        graph_pred = torch_from_preds(graph, predictions)
        nc_gap, vola, volb, cut,ic,id = normalized_cut(graph_pred)
        ncuts.append(nc_gap)
        vols.append((vola, volb))
        preds.append(predictions)
        cuts.append(cut)
        ias.append(ic)
        ibs.append(id)
    min_nc = np.argmin(ncuts)
    min_c = np.argmin(cuts)
    return ncuts[min_c], vols[min_c], preds[min_c], cuts[min_c], t1 + t1_loop,ias[min_c],ibs[min_c],ia,ib


def torch_from_preds(graph, preds):
    graph_torch = graph
    graph_torch.x = preds
    return graph_torch


def volumes(graph):
    ia = torch.where(graph.x == torch.tensor(0.))[0]
    ib = torch.where(graph.x == torch.tensor(1.))[0]
    degs = degree(
    graph.edge_index[0],
    num_nodes=graph.x.size(0),
     dtype=torch.uint8) # degs the degree of all nodes (tensor)
    da = torch.sum(degs[ia]).detach().item() # total degree of partition a
    db = torch.sum(degs[ib]).detach().item() # total degree of partition complementary set a
    cut = torch.sum(graph.x[graph.edge_index[0]] !=
                    graph.x[graph.edge_index[1]]).detach().item() / 2
    return cut, da, db,ia,ib

def cut(graph):
    cut = torch.sum((graph.x[graph.edge_index[0],
    :2] != graph.x[graph.edge_index[1],
     :2]).all(axis=-1)).detach().item() / 2
    return cut

def normalized_cut(graph):
    c, dA, dB,ia,ib = volumes(graph)
    if dA == 0 or dB == 0:
        return 2, 0, dA if dA != 0 else dB, c,ia,ib
    else:
        return c / dA + c / dB, dA, dB, c,ia,ib

def pymetis_baseline(data):
    adjacent_list = from_adjacency_matrix_to_adjacency_list(data)
    n_cut,membership = pymetis.part_graph(2, adjacency=adjacent_list)
    balance = list(Counter(membership).values())
    return n_cut,balance

def remove_file_or_folder(path,mode='file'):
    if mode=='file':
        print('delete file')
        try:
            os.remove(path)
        except Exception as result:
            print(str(result)+' , '+'Deleting fails')
    elif mode=='folder':
        print('delete empty folder')
        try:
            os.rmdir(path)
        except Exception as result:
            print(str(result)+' , '+'Deleting fails')
    elif mode=='all in folder':
        print('delete all in folder')
        try:
            shutil.rmtree(path)
        except Exception as result:
            print(str(result)+' , '+'Deleting fails')
    else:
        print('no mode')

def print_message(data,mode,config):
    if mode=='train' and data=='all':
        print('')
        print('Number of SuiteSparse of train: ',int(len(config.loader)/3))

    else :
        for d in config.loader:
            graph = SparseTensor(row=d.edge_index[0],col=d.edge_index[1],value=torch.ones_like(d.edge_index[0],dtype=torch.float64))
            config.baseline,config.balance = pymetis_baseline(graph)
            print('data: ',graph)
            print('')
            print('The number of nodes: ',graph.storage._sparse_sizes[0])
            print('The number of edges: ',int(graph.storage._row.shape[0]/2))
            print('')
            print('metis cuts: ',config.baseline)
            print('metis balance: ',config.balance)
    print('')
    print('model: ',config.model)