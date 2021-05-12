
import torchvision 
import pdb
import os
import torchvision.transforms as transforms
from dataset import *
import torch
import torch.nn as nn
from utils import *
import time
import numpy as np
from imageretrievalnet import *
from loss import *
import time
import argparse
from MAP import *
from  collections import OrderedDict
from resnet50 import resnet50
import cv2
import shutil

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

def copystatedict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict




parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')


parser.add_argument('--cls-num', default=101, type=int,
                    metavar='N', help='class number')

parser.add_argument('--mul-cls-num', default=174, type=int,
                    metavar='N', help='ingradient class number')
                    
parser.add_argument('--epoch', default=100, type=int,
                    metavar='N', help='epochs')
parser.add_argument('--bs', default=64, type=int,
                    metavar='N', help='batch size')
parser.add_argument('--imsize', default=362, type=int,
                    metavar='N', help='image size')
parser.add_argument('--lr', default=1e-4, type=float,
                    metavar='N', help='learning rate')
parser.add_argument('--dataset', default='ChineseFoodNet', type=str,
                     help='dataset name')
parser.add_argument('--dataroot', default='/workdir/lizhuo/dataset/Food_API_test_dataset_clean', type=str,
                     help='dataset path')
parser.add_argument('--net', default='resnet101', type=str,
                     help='network')
parser.add_argument('--batch-p', default=8, type=int,
                    metavar='N', help='class per batch')
parser.add_argument('--batch-k', default=4, type=int,
                    metavar='N', help='images per class')

parser.add_argument('--resume', default=None, type=str,
                     help='checkpoint path')

parser.add_argument('--loss', default='cross', type=str,
                    help='loss function')

parser.add_argument('--pool', default='gem', type=str,
                    help='loss function')

parser.add_argument('--test',action='store_true',
                    help='test before training')

parser.add_argument('--mg',action='store_true',
                     help='multi-gpu')

parser.add_argument('--cls-only',action='store_true',
                    help='only cross loss')

parser.add_argument('--dim', default=512, type=int,
                    metavar='N', help='embedding dim')
import faiss

import numpy as np




def test(test_loader_q, test_loader_g, model, epoch):

    print('>> Evaluating network on test datasets...')

    batch_time = AverageMeter()

    data_time = AverageMeter()
    ngpus = faiss.get_num_gpus()

    d = 1792
    
    gpu_index = cpu_index = faiss.IndexFlatL2(d)
    #gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
    #    cpu_index
    #)
   
    end = time.time()

    model.eval()

    dataset_q = []
    
    code_q = []

    for step, (x, code) in enumerate(test_loader_q):
        

        code_q.extend(code)
        batch_time.update(time.time() - end)

        end = time.time()

        x = x.cuda()

        x = x.contiguous()

        with torch.no_grad():
            vec, _ = model(x)

        dataset_q.extend(vec.unsqueeze(0))


        if step % 100 == 0:

            print('>> Test: [{0}][{1}/{2}]\t'

                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'

                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'

                .format(

                epoch+1, step+1, len(test_loader_q), batch_time=batch_time,

                data_time=data_time))



    dataset_q = torch.cat(dataset_q, dim = 0)
    dataset_q = dataset_q.cpu().numpy()
    dataset_g = []
    code_g = []

    for step, (x, code) in enumerate(test_loader_g):

        batch_time.update(time.time() - end)

        end = time.time()
        
        code_g.extend(code)        

        x = x.cuda()

        x = x.contiguous()



        with torch.no_grad():

            vec, _ = model(x)




        gpu_index.add(vec.cpu().numpy())



        if step % 100 == 0 and step != 0:

            print('>> Test: [{0}][{1}/{2}]\t'

                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'

                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'

                .format(

                epoch+1, step+1, len(test_loader_g), batch_time=batch_time,

                data_time=data_time))
            # break
            





    index_path = ''


    print('#images in database:',gpu_index.ntotal)
    k = 10                          # we want to see 4 nearest neighbors
    D, I = gpu_index.search(dataset_q, k) # actual search
    # pdb.set_trace()
    with open('submit_v2_test.csv','w')  as f:
        for i in range(2000):
            
            f.write(code_q[i] + ',')
            path = '/data1/sjj/ePruduct_dataset/query_part1/'+code_q[i]+'.JPEG'
            img_q = cv2.imread(path)
            img_q = cv2.resize(img_q, (224,224))
            img_r = []
            for j in range(10):
                f.write(code_g[I[i][j]] + ' ')
                path = '/data1/sjj/ePruduct_dataset/index/'+code_g[I[i][j]]+'.JPEG'
                img = cv2.imread(path)
                img = cv2.resize(img, (224,224))
                font = cv2.FONT_HERSHEY_SIMPLEX
                img = cv2.putText(img, str(D[i][j]), (112, 112 ), font, 1.2, (255 , 0, 0 ), 2)
                img_r += [img]
            
            # # pdb.set_trace()
            for j in range(10):
                img_q = np.concatenate((img_q, img_r[j]), axis = 1 )
            
            save_path = os.path.join('visual', code_q[i]+'.JPEG')
            cv2.imwrite(save_path, img_q)
            # f.write(',')
            # for j in range(10):
            #     f.write(str(D[i][j]) + ' ')
            f.write('\r\n')


def test_single_dataset(model):



    BATCH_SIZE = 512
    
    

 

    ####################################################
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),normalize
    ])

    img_root = '/data1/sjj/ePruduct_dataset'
    
    file_list = os.path.join(img_root, 'query_part1.csv')

    test_dataset_q = ImagesForTest(img_root, file_list, transform=transform)  
    test_loader_q = torch.utils.data.DataLoader(
        test_dataset_q, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=8, pin_memory=True, sampler=None,
    )
    
    file_list = os.path.join(img_root, 'index.csv')
    test_dataset_g = ImagesForTest(img_root, file_list,  transform=transform)  
    test_loader_g = torch.utils.data.DataLoader(
        test_dataset_g, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=8, pin_memory=True, sampler=None,
    )
    
    test(test_loader_q, test_loader_g, model, -1)



def main():

    global args
    args = parser.parse_args()

    

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

    model = image_net(net_name,args).cuda()

    checkpoint = torch.load(args.resume)
    if isinstance(checkpoint,DataParallel):
        checkpoint = checkpoint.module.state_dict()
    model.load_state_dict(checkpoint)
    if args.mg:
        model=nn.DataParallel(model,device_ids=[0,1,2,3]) 
    ####################################################
    if args.resume != None:
        checkpoint = torch.load(args.resume)
        if 'state_dict' in checkpoint:
           checkpoint = checkpoint['state_dict']
        checkpoint = copystatedict(checkpoint)
        model.load_state_dict(checkpoint)
    
    
    print(test_all_dataset(model))
    exit()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),normalize
    ])
    img_root = os.path.join(args.dataroot,args.dataset,'test_imgs')
    file_list = os.path.join(args.dataroot,args.dataset,'test_single_info','query_imgs')
    test_dataset_q = ImagesForTest(img_root, file_list, image_size, transform=transform)  
    test_loader_q = torch.utils.data.DataLoader(
        test_dataset_q, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True, sampler=None,
    )
    

    file_list = os.path.join(args.dataroot,args.dataset,'test_single_info','database_imgs')
    test_dataset_g = ImagesForTest(img_root, file_list, image_size, transform=transform)  
    test_loader_g = torch.utils.data.DataLoader(
        test_dataset_g, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True, sampler=None,
    )
    
    AP, precision, mAP, recall = test(test_loader_q, test_loader_g, model, -1)
    
    print("AP:{0}\tprecision:{1}\tmAP:{2}\trecall:{3}".format(AP,precision,mAP,recall))
if __name__=='__main__':
    main()
