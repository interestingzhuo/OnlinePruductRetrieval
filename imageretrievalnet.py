
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
from efficientnet_pytorch import EfficientNet

import timm


pool_dic = {
"GeM":GeM,
"SPoC":SPoC,
"MAC":MAC,
"RMAC":RMAC,
"GeMmp":GeMmp
 }

class ImageRetrievaleffNet(nn.Module):
    def __init__(self, net):

        super(ImageRetrievaleffNet, self).__init__()
        self.net = net
        self.norm = L2N()



    def forward(self, x, test=False):

        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = x.size(0)
        # Convolution layers
        x = self.net.extract_features(x)

        # Pooling and final linear layer
        x = self.net._avg_pooling(x)
        o = x
        x = x.view(bs, -1)
        x = self.net._dropout(x)
        x = self.net._fc(x)
        o = self.norm(o).squeeze(-1).squeeze(-1)

        return o, x


class ImageRetrievalresNet(nn.Module):
    def __init__(self, features,fc_cls,pool):
        
        super(ImageRetrievalresNet, self).__init__()
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
class VITImageRetrievalNet(nn.Module):
    def __init__(self, net):

        super(VITImageRetrievalNet, self).__init__()
        self.net = net
        self.norm = L2N()

    def forward(self, x, test=False):
        o = self.net.forward_features(x)
        x = self.net.head(o)
        o = self.norm(o).squeeze(-1).squeeze(-1)
        return o, x

def image_net(net_name,opt):
    if net_name == 'resnet101':
        net = torchvision.models.resnet101(pretrained=True)
    elif net_name == 'resnet50':
        net = torchvision.models.resnet50(pretrained=True)
    elif 'vit' in net_name:
        net = timm.create_model(net_name, pretrained = True)
        #x = self.forward_features(x)
        #x = self.head(x)
        net.head = nn.Linear(net.embed_dim, opt.cls_num)
        return VITImageRetrievalNet(net)
         
    elif 'efficient' in net_name:
        net = EfficientNet.from_pretrained(net_name, num_classes=opt.cls_num)
        return ImageRetrievaleffNet(net)
    else:
        raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))

    net.fc = nn.Linear(in_features=2048, out_features=opt.cls_num, bias=True)


    features = list(net.children())[:-2]
    fc_cls = net.fc
    if "R-" in opt.pool:
        if opt.pool == 'R-ori':
            pool = net.avgpool
        else:
            pool =  pool_dic[opt.pool[2:]]()
        pool = Rpool(pool)
    else:
        if opt.pool == 'ori':
            pool = net.avgpool
        else:
           pool = pool_dic[opt.pool]()
    return ImageRetrievalresNet(features,fc_cls,pool)


