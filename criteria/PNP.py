
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer
import pdb

"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False




class Criterion(torch.nn.Module):
    """PyTorch implementation of the Smooth-AP loss.
    implementation of the Smooth-AP loss. Takes as input the mini-batch of CNN-produced feature embeddings and returns
    the value of the Smooth-AP loss. The mini-batch must be formed of a defined number of classes. Each class must
    have the same number of instances represented in the mini-batch and must be ordered sequentially by class.
    e.g. the labels for a mini-batch with batch size 9, and 3 represented classes (A,B,C) must look like:
        labels = ( A, A, A, B, B, B, C, C, C)
    (the order of the classes however does not matter)
    For each instance in the mini-batch, the loss computes the Smooth-AP when it is used as the query and the rest of the
    mini-batch is used as the retrieval set. The positive set is formed of the other instances in the batch from the
    same class. The loss returns the average Smooth-AP across all instances in the mini-batch.
    Args:
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function. A low value of the temperature
            results in a steep sigmoid, that tightly approximates the heaviside step function in the ranking function.
        batch_size : int
            the batch size being used during training.
        num_id : int
            the number of different classes that are represented in the batch.
        feat_dims : int
            the dimension of the input feature embeddings
    Shape:
        - Input (batch): (batch_size, feat_dims) (must be a cuda torch float tensor)
        - Output: scalar
    Examples::
        >>> loss = SmoothAP(0.01, 60, 6, 256)
        >>> input = torch.randn(60, 256, requires_grad=True).cuda()
        >>> output = loss(input)
        >>> output.backward()
    """
    def __init__(self, opt, batchminer):
        super(Criterion, self).__init__()

        assert(opt.bs%opt.samples_per_class==0)
        self.b = opt.b
        self.alpha = opt.alpha
        self.anneal = opt.anneal
        self.relax = opt.relax
        self.batch_size = opt.bs
        self.margin = opt.margin
        self.num_id = int(opt.bs/opt.samples_per_class)
        self.samples_per_class = opt.samples_per_class
        self.feat_dims = opt.embed_dim
        self.batchminer = batchminer
        self.name           = 'rank'
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM
        mask = 1.0 - torch.eye(self.batch_size)
        for i in range(self.num_id):
            mask[i*(self.samples_per_class):(i+1)*(self.samples_per_class),i*(self.samples_per_class):(i+1)*(self.samples_per_class)] = 0
        
        self.mask = mask.unsqueeze(dim=0).repeat(self.batch_size, 1, 1).cuda()
        self.margin_mask = torch.eye(self.batch_size).cuda()
        for i in range(self.num_id):
            self.margin_mask[i*(self.samples_per_class):(i+1)*(self.samples_per_class),i*(self.samples_per_class):(i+1)*(self.samples_per_class)] = self.margin
    def forward(self, batch, labels, **kwargs):
        if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
        # compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        sim_all = compute_aff(batch)
        sim_all = sim_all - self.margin_mask
        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, self.batch_size, 1)
        # compute the difference matrix
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        # pass through the sigmoid and ignores the relevance score of the query to itself
        sim_sg = sigmoid(sim_diff, temp=self.anneal) * self.mask
        # compute the rankings,all batch
        sim_all_rk = torch.sum(sim_sg, dim=-1) 
        if self.relax == 'log_neg':
            sim_all_rk = torch.log(1+sim_all_rk)
        elif self.relax == 'reci':
            sim_all_rk = 1/(1+sim_all_rk)**(self.alpha)

        #elif self.relax == 'reci_2':
        #    sim_all_rk = 1/((1+sim_all_rk)**2)

        elif self.relax == 'log_pos':
            sim_all_rk = (1+sim_all_rk)*torch.log(1+sim_all_rk)
            
        elif self.relax == 'b_1':
            b = self.b
            sim_all_rk = 1/b**2 * (b*sim_all_rk-torch.log(1+b*sim_all_rk))
        elif self.relax == 'ori':
            pass
        else:
                raise Exception('Relaxation <{}> not available!'.format(self.relax))
        
        
        # sum the values of the Smooth-AP for all instances in the mini-batch
        loss = torch.zeros(1).cuda()
        group = int(self.batch_size / self.num_id)
        
        

        for ind in range(self.num_id):
            
            neg_divide = torch.sum(sim_all_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)])
            '''
            if self.relax == 'log_neg':
                neg_divide = torch.log(1+neg_divide)
            elif self.relax == 'reci':
                neg_divide = 1/(1+neg_divide)

            elif self.relax == 'log_pos':
                neg_divide = (1+neg_divide)*torch.log(1+neg_divide)
            
            elif self.relax == 'b_1':
                b = 2
                neg_divide = 1/b**2 * (b*neg_divide-torch.log(1+b*neg_divide))
            elif self.relax == 'ori':
                pass
            else:
                raise Exception('Relaxation <{}> not available!'.format(self.relax))
            '''    
            #松弛
            '''
            if self.relax == 'log':
                neg_divide = 1/(1+neg_divide)
            elif self.relax == 'arctan':
                neg_divide = neg_divide*torch.atan(neg_divide) -1/2*torch.log(1+neg_divide**2);
            else:
                raise Exception('Relaxation <{}> not available!'.format(self.relax))
            '''
            loss = loss + ((neg_divide / group) / self.batch_size)
        '''
        if self.relax == 'log':
            loss = (1+loss)*(-torch.log(loss))
        '''
        if self.relax[:4] == 'reci':
            return 1 - loss
        else:
            return loss


def sigmoid(tensor, temp=1.0):
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


def compute_aff(x):
    """computes the affinity matrix between an input vector and itself"""
    return torch.mm(x, x.t())
