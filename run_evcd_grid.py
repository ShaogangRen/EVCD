"""
grid search
"""
import cdt.causality.pairwise 
import cdt.causality.graph
import torch
import networkx as nx
from cdt.data import load_dataset
import numpy as np
from EVCD import FlowGraph
from multiprocessing import Pool
import os 
from loguru import logger

import random
seed = 5
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def train_flow_evcd(param):
    
    dataset, flow_depth, lr, max_epoch, w_init_sigma = param 
    
    LocalProcRandGen = np.random.RandomState()

    ### set GPU device range
    cuda_device = LocalProcRandGen.choice([0, 1])
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
    
    logger.info("Start {0}.{1}.{2}.{3}.{4}.log".format(dataset, flow_depth, lr, max_epoch, w_init_sigma))
    
    obj = FlowGraph(max_epoch = max_epoch, lr =lr, w_init_sigma=w_init_sigma, flow_depth=flow_depth)
    
    if  dataset.startswith("dream4"):
        data, graph = load_dataset(dataset)
        
        output = obj.orient_graph(data, nx.Graph(graph))     
        
        total_edge_num = 0
        match_edge_num = 0
        for n1, n2 in graph.edges():
            total_edge_num += 1
            if output.has_edge(n1, n2):
                match_edge_num += 1
                
        acc = match_edge_num/ total_edge_num
    else:
        
        data, labels = load_dataset(dataset)
            
        output = obj.predict(data)
        
        try:
            output = np.array(output)
        except:
            print(type(output))
            raise 

        print(labels.Target.shape)
        pair_n = output.shape[0]

        pred = output.reshape((pair_n,))

        pred[pred>0.0] = 1
        pred[pred<=0.0] = -1

        correct = np.zeros(pair_n)

        print(pred.shape)

        correct[pred == labels.Target] = 1
        acc = 1.0*sum(correct)/pair_n
        
    print('accuracy = {}'.format(acc))
    
    with open("{0}.{1}.{2}.{3}.{4}.log".format(dataset, flow_depth, lr, max_epoch, w_init_sigma), "w") as f:
        f.write("{0}.{1}.{2}.{3}.{4}\n".format(dataset, flow_depth, lr, max_epoch, w_init_sigma))
        f.write(str(acc))

           
            
def grid_search():
                
    flow_depth = [2]
    lr = [0.001]
    max_epoch = [800,1500]
    w_init_sigma = [0.1] #0.01,
    #data_sets = ["tuebingen", "dream4-1", "dream4-2", "dream4-3", "dream4-4", "dream4-5"]
    data_sets = ["tuebingen"]

    import itertools

    param_grid = itertools.product(data_sets, flow_depth, lr, max_epoch, w_init_sigma)

    with Pool(12) as pool:
        result = pool.map(train_flow_evcd, param_grid)
                
                
if __name__ == "__main__":
                
    grid_search()
        
