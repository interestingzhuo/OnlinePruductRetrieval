
import torchvision 
import pdb
import os
import torchvision.transforms as transforms
from dataset import ImagesForCls
import torch
import torch.nn as nn
from utils import *
import time
import numpy as np
from imageretrievalnet import *
from loss import *
import time
import argparse
from torch.nn.parallel.data_parallel import DataParallel
from MAP import *


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')


parser.add_argument('--cls-num', default=20000, type=int,
                    metavar='N', help='class number')

parser.add_argument('--epoch', default=100, type=int,
                    metavar='N', help='epochs')
parser.add_argument('--bs', default=64, type=int,
                    metavar='N', help='batch size')
parser.add_argument('--imsize', default=362, type=int,
                    metavar='N', help='image size')
parser.add_argument('--lr', default=1e-2, type=float,
                    metavar='N', help='learning rate')
parser.add_argument('--dataset', default='mt20000', type=str,
                     help='dataset name')
parser.add_argument('--dataroot', default='/workdir/lizhuo/dataset', type=str,
                     help='dataset path')

parser.add_argument('--resume', default=None, type=str,
                      help='checkpoint path')
parser.add_argument('--net', default='resnet50', type=str,
                     help='network')
parser.add_argument('--loss', default='cross', type=str,
                    help='loss function')

parser.add_argument('--test',action='store_true',
                    help='test before training')

parser.add_argument('--mg',action='store_true',
                     help='multi-gpu')
parser.add_argument('--workers', default=8, type=int,
                     metavar='N', help='number works')

def main():

    global args
    args = parser.parse_args()
    #model = torchvision.models.resnet50(pretrained=True)
    
    #model.fc = nn.Linear(in_features=2048, out_features=args.cls_num, bias=True)
    model = image_net('resnet50',20000,174)
    if args.resume != None:
        print('loading checkpoint from:',args.resume)
        checkpoint = torch.load(args.resume)
        if isinstance(checkpoint,DataParallel):
            model.load_state_dict(checkpoint.module.state_dict())
        else:
            model.load_state_dict(checkpoint)

    model = model.cuda()

    

    EPOCHS = args.epoch

    BATCH_SIZE = args.bs
    
    image_size = args.imsize
    
    lr = args.lr

    dataset = args.dataset
 
    root = args.dataroot
    
    ann_folder = os.path.join(root, dataset, 'retrieval_dict')
    
    imgs_root = os.path.join(root, dataset, 'images')

    net_name = args.net

    cls_num = args.cls_num

     
    

    if args.mg:
        model=nn.DataParallel(model,device_ids=[0,1,2,3]) 
    ####################################################
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),normalize
    ])
    

    
    criterion =  nn.CrossEntropyLoss()

    localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    d = localtime+'_'+args.loss;
    

    directory = os.path.join(dataset,d)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    train_folder = 'MTCV_merge_to_dishname_20191108_stage2_sku_2w_imgs_320_changemode_train_changemode'
    train_dataset = ImagesForCls(os.path.join(imgs_root,train_folder), image_size,transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None,
    )

    test_folder = 'MTCV_merge_to_dishname_20191108_stage2_sku_2w_imgs_320_changemode_val_changemode'
    test_dataset = ImagesForCls(os.path.join(imgs_root,test_folder), image_size,transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay=1e-5, momentum=0.9)
    Logger_file = os.path.join(directory,"log.txt")
    
    if args.test:
        AP, precision, mAP, recall = test(test_loader, model, -1)
        print('AP:',AP)
        print('precision:',precision)
        print('mAP:',mAP)
        with open(Logger_file,'a') as f:
            f.write("epoch:{}\tAP@m:{}\tPrecision:{}\tmAP:{}\trecall:{}".format(-1,AP,precision,mAP,recall))
    for epoch in range(EPOCHS):
         
        train(train_loader,model,epoch,criterion,optimizer,args)
        torch.cuda.empty_cache()
        
        AP,precision,mAP, recall = test(test_loader, model, epoch)
        print('AP:',AP)
        print('precision:',precision)
        print('mAP:',mAP)
        with open(Logger_file,'a') as f:
            f.write("epoch:{}\tAP@m:{}\tPrecision:{}\tmAP:{}\trecall:{}\n".format(epoch,AP,precision,mAP,recall))
        path = os.path.join(directory,'model_epoch_{}.pth'.format(epoch))
        torch.save(model,path)
        if epoch % 3 == 1:
            optimizer.param_groups[0]['lr']/=2

def train(train_loader,model,epoch,criterion, optimizer,args, criterion_metric=None):
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    end = time.time()
    for step, (x, y) in enumerate(train_loader):
        batch_time.update(time.time() - end)
        end = time.time()
        
        x = x.cuda()
        _,out = model(x)
        

        loss = criterion(out, y.cuda())#分类损失

        train_loss.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 1 and step != 0:
            print('>> Train: [{0}][{1}/{2}]\t lr:{3:.3f}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {train_loss.val:.3f} ({train_loss.avg:.3f})\t'
                .format(
                epoch+1, step+1, len(train_loader),optimizer.param_groups[0]['lr'], batch_time=batch_time,
                train_loss=train_loss))



def test(test_loader, model, epoch):
    print('>> Evaluating network on test datasets...')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.eval()
    ap_meter = AveragePrecisionMeter(False)
    precision = PrecisionMeter(False)
    output = []
    clusters = []
    for step, (x, lbl) in enumerate(test_loader):
        batch_time.update(time.time() - end)
        end = time.time()
        x = x.cuda()
        x = x.contiguous()
        
        with torch.no_grad():
            _,out = model(x)
        output.extend(out.cpu().numpy())
        clusters.extend(lbl.numpy())
        #precision.add(out_cls.data,lbl)

        
        if step % 100 == 0:
            print('>> Test: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                epoch+1, step+1, len(test_loader), batch_time=batch_time,
                data_time=data_time))
    output = np.array(output)
    output = output.argmax(axis = 1)
    clusters = np.array(clusters)
    precision =  clusters== output
    precision = sum(precision)/len(precision)
    mAP = 0
    recall = 0
    return 0, precision, mAP, recall
if __name__=='__main__':
    main()
