import torch
import pdb
import os
import numpy as np
import sklearn.metrics.pairwise as skp
import gc
import time
from multiprocessing import Process,Lock

#由并发变成了串行,牺牲了运行效率,但避免了竞争
from multiprocessing import Process,Lock
'''
def work(lock):
    lock.acquire()
    print('%s is running' %os.getpid())
    time.sleep(2)
    print('%s is done' %os.getpid())
    lock.release()
if __name__ == '__main__':
    lock=Lock()
    for i in range(3):
        p=Process(target=work,args=(lock,))
        p.start()
'''
def comp_Ap(list_retrieval):
    m=0;
    Ap=0.;
    for i in range(len(list_retrieval)):
        if list_retrieval[i]:
            m+=1
            Ap+=m/(i+1)
    return Ap/m


def comp_AP_k(binary,top):
    m = 0
    Ap = 0
    for i in range(top):
        if binary[i]:
           m += 1
           Ap += m/(i+1)
    #if m == 0:
    #    return 0
    return Ap/top

def comp_rc(binary,top):
    r = 0;
    for i in range(1,top+1):
        if binary[i]:
	        r = 1 
	        break
    return r 

def comp_MAp(ranks,clusters,similarity):
    Ap=[0]*3;
    recall = [0]*3;
    top = [1,5,10]


    for i in range(ranks.shape[0]):
        binary=[clusters[i]==clusters[j] for j in ranks[i]]
        for j in range(3):
            r = comp_rc(binary,top[j])
            a = comp_AP_k(binary,top[j])
            recall[j] += r
            Ap[j] += a
    recall=[r/ranks.shape[0] for r in recall] 
    Ap=[a/ranks.shape[0] for a in Ap]
    return Ap,recall

def comp_MAp_k(ranks,clusters_q,clusters_g):
    
    Ap=[0]*3;
    recall = [0]*3;
    top = [1,3,5]


    for i in range(ranks.shape[0]):
        binary=[clusters_q[i]==clusters_g[j] for j in ranks[i]]
        for j in range(3):
            r = comp_rc(binary,top[j])
            a = comp_AP_k(binary,top[j])
            recall[j] += r
            Ap[j] += a
    recall=[r/ranks.shape[0] for r in recall] 
    Ap=[a/ranks.shape[0] for a in Ap]
    return Ap,recall


def Test(dataset,clusters):
    st = time.time()
    pdb.set_trace()
    similarity = torch.mm(dataset, dataset.t())
    print('distance_time:',time.time() - st)
    st = time.time()
    ranks = torch.argsort(similarity,dim=1)
    print('rank_time:',time.time() - st)
    MAp,recall = comp_MAp(ranks,clusters,similarity);
    return MAp,recall 
def Test(dataset_q,dataset_g,cluster_q,cluster_g):
    st = time.time()
    similarity = torch.mm(dataset_q, dataset_g.t())
    print('distance_time:',time.time() - st)
    st = time.time()
    ranks = torch.argsort(similarity,dim=1,descending=True)
    print('rank_time:',time.time() - st)
    MAp,recall = comp_MAp_k(ranks,cluster_q,cluster_g);
    return MAp,recall 
