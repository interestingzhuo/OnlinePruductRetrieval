
import os
import pdb

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.parallel.data_parallel import DataParallel

from pooling import *

from normalization import *





pool_dic = {
"GeM":GeM,
"SPoC":SPoC,
"MAC":MAC,
"RMAC":RMAC,
"GeMmp":GeMmp
 }


class ImageRetrievalNet(nn.Module):
    def __init__(self, features,fc_cls,pool):
        
        super(ImageRetrievalNet, self).__init__()
        self.features = nn.Sequential(*features)
        self.pool = pool
        self.norm = L2N()
        if type(fc_cls)==list: 
          self.fc_cls = nn.Sequential(*fc_cls)   
        else:
          self.fc_cls=fc_cls
        
        

    

    
    def forward(self, x, test=False):
        o = self.features(x)
        

        o = self.pool(o)

        cls = self.fc_cls(o.squeeze())

            

        o = self.norm(o).squeeze(-1).squeeze(-1)
        
        return o, cls

def image_net(net_name,opt):
    if net_name == 'resnet101':
        net = torchvision.models.resnet101(pretrained=True)
    elif net_name == 'resnet50':
        net = torchvision.models.resnet50(pretrained=True)
    else:
         raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))

    net.fc = nn.Linear(in_features=2048, out_features=opt.cls_num, bias=True)


    features = list(net.children())[:-2]
    fc_cls = net.fc
    if "R-" in opt.pool:
         pool =  pool_dic[opt.pool[2:]]()
         pool = Rpool(pool)
    elif opt.pool == 'ori':
        pool = net.avgpool
    else:
         pool = pool_dic[opt.pool]()
    return ImageRetrievalNet(features,fc_cls,pool)
