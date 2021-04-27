import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
def normalize(A , symmetric=True):
    # A = A+I
    A = A + torch.eye(A.size(0))
    # 所有节点的度
    d = A.sum(1)
    if symmetric:
        #D = D^-1/2
        D = torch.diag(torch.pow(d , -0.5))
        return D.mm(A).mm(D)
    else :
        # D=D^-1
        D =torch.diag(torch.pow(d,-1))
        return D.mm(A)


class GCN(nn.Module):
    def __init__(self , A, dim_in , dim_out):
        super(GCN,self).__init__()
        self.fc1 = nn.Linear(dim_in ,dim_in,bias=True)
        self.fc2 = nn.Linear(dim_in,dim_in//2,bias=True)
        self.fc3 = nn.Linear(dim_in//2,dim_out,bias=True)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.A = normalize(torch.from_numpy(A),True).cuda()

    def forward(self,X):
        '''
        计算三层gcn
        '''
        #建立相似度矩阵
        X = X.reshape((-1,self.dim_in))
        X = F.relu(self.fc1(self.A.mm(X)))
        X = F.relu(self.fc2(self.A.mm(X)))
        X = self.fc3(self.A.mm(X))
        kernel_size = X.size(0)
        X = X.T
        X = X.unsqueeze(dim = 0)
        X = F.avg_pool1d(X, kernel_size)
        return X


