import torch
import pdb
import os
import numpy as np
import faiss
import time
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
def comp_MAp(ranks,clusters,similarity):
    Ap=[0]*3;
    recall = [0]*3;
    top = [1,5,10]


    for i in range(ranks.shape[0]):
        binary=[clusters[i]==clusters[j] for j in ranks[i]]
        for j in range(3):
            a = comp_AP_k(binary,top[j])
            Ap[j] += a
    Ap=[a/ranks.shape[0] for a in Ap]
    return Ap,recall

def comp_MAp_k(ranks,clusters_q,clusters_g):
    
    Ap=[0]*3;
    recall = [0]*3;
    top = [1,5,10]


    for i in range(ranks.shape[0]):
        binary=[clusters_q[i]==clusters_g[j] for j in ranks[i][1:]]
        for j in range(3):
            a = comp_AP_k(binary,top[j])
            Ap[j] += a
    Ap=[a/ranks.shape[0] for a in Ap]
    return Ap,recall


def Test_mAP(dataset,clusters):
    st = time.time()
    similarity = torch.mm(dataset, dataset.t())
    print('distance_time:',time.time() - st)
    st = time.time()
    ranks = torch.topk(similarity,11,dim=1).indices
    print('rank_time:',time.time() - st)
    MAp,recall = comp_MAp_k(ranks,clusters,clusters);
    
    return MAp,recall    

    #ranks = torch.argsort(similarity,dim=1)
    #print('rank_time:',time.time() - st)
    #MAp,recall = comp_MAp(ranks,clusters,similarity);
    #return MAp,recall 


def Test(dataset_q,dataset_g,cluster_q,cluster_g):
    st = time.time()
    similarity = torch.mm(dataset_q, dataset_g.t())
    print('distance_time:',time.time() - st)
    st = time.time()
    ranks = torch.topk(similarity,5,dim=1).indices
    #ranks = torch.argsort(similarity,dim=1,descending=True)
    print('rank_time:',time.time() - st)
    MAp,recall = comp_MAp_k(ranks,cluster_q,cluster_g);
    return MAp,recall
