import numpy as np
import torch
import torch.utils.data as data
import os
import pdb
import json
import cv2
import torchvision.transforms as transforms
import random
from PIL import Image


class ImagesForCls_list(data.Dataset):

    def __init__(self,img_root, file_list, imsize=224, bbxs=None, transform=None,is_validation=False):


        self.images_fn, self.clusters = self.get_imgs(file_list,img_root)
        self.imsize = imsize
        self.transform = transform

        self.is_validation = is_validation
        self.f_norm = normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        transf_list = []

        self.crop_size = crop_im_size = 224

        #############
        self.normal_transform = []
        if not self.is_validation:
                self.normal_transform.extend([transforms.RandomResizedCrop(size=crop_im_size), self.transform, transforms.RandomHorizontalFlip(0.5)])
                # self.normal_transform.extend([transforms.RandomResizedCrop(size=crop_im_size), self.transform, transforms.RandomGrayscale(p=0.2),
                #                               transforms.ColorJitter(0.2, 0.2, 0.2, 0.2), transforms.RandomHorizontalFlip(0.5)])
        else:
            self.normal_transform.extend([transforms.Resize(256), transforms.CenterCrop(crop_im_size)])
        self.normal_transform.extend([transforms.ToTensor(), normalize])
        self.normal_transform = transforms.Compose(self.normal_transform)
    def get_imgs(self, train_list, img_root):
        with open(train_list) as f:
            lines = f.readlines()
        images = [item.strip().split()[0] for item in lines]
        images = [os.path.join(img_root,item) for item in images]
        clusters = [int(item.strip().split()[1]) for item in lines]
        return images,np.array(clusters)

    def ensure_3dim(self, img):
         if len(img.size)==2:
             img = img.convert('RGB')
         return img
    def __getitem__(self, index):
        img = self.ensure_3dim(Image.open(self.images_fn[index]))
        img = self.normal_transform(img)
        return img, self.clusters[index]

    def __len__(self):
        return len(self.images_fn)




class ImagesForCls_list_p(data.Dataset):

    def __init__(self,file_list, imsize=224, bbxs=None, transform=None,is_validation=False):


        self.images_fn, self.clusters = self.get_imgs(file_list)
        self.imsize = imsize
        self.transform = transform

        self.is_validation = is_validation
        self.f_norm = normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        transf_list = []

        self.crop_size = crop_im_size = 224

        #############
        self.normal_transform = []
        if not self.is_validation:
                self.normal_transform.extend([transforms.RandomResizedCrop(size=crop_im_size), transforms.RandomGrayscale(p=0.2),
                                              transforms.ColorJitter(0.2, 0.2, 0.2, 0.2), transforms.RandomHorizontalFlip(0.5)])
        else:
            self.normal_transform.extend([transforms.Resize(256), transforms.CenterCrop(crop_im_size)])
        self.normal_transform.extend([transforms.ToTensor(), normalize])
        self.normal_transform = transforms.Compose(self.normal_transform)
    def get_imgs(self,train_list):
        with open(train_list) as f:
            lines = f.readlines()
        images = [item.strip().split()[0] for item in lines]
        clusters = [int(item.strip().split()[1]) for item in lines]
        return images,np.array(clusters)

    def ensure_3dim(self, img):
         if len(img.size)==2:
             img = img.convert('RGB')
         return img
    def __getitem__(self, index):
        img = self.ensure_3dim(Image.open(self.images_fn[index]))
        img = self.normal_transform(img)
        return img, self.clusters[index]

    def __len__(self):
        return len(self.images_fn)



class ImagesForCls(data.Dataset):

    def __init__(self, ims_root, imsize=224, bbxs=None, transform=None,is_validation=False):


        self.root = ims_root
        self.images_fn, self.clusters = self.get_imgs(ims_root)
        self.imsize = imsize
        self.transform = transform

        self.is_validation = is_validation
        self.f_norm = normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        transf_list = []

        self.crop_size = crop_im_size = 224

        #############
        self.normal_transform = []
        if not self.is_validation:
                self.normal_transform.extend([transforms.RandomResizedCrop(size=crop_im_size), transforms.RandomGrayscale(p=0.2),
                                              transforms.ColorJitter(0.2, 0.2, 0.2, 0.2), transforms.RandomHorizontalFlip(0.5)])
        else:
            self.normal_transform.extend([transforms.Resize(256), transforms.CenterCrop(crop_im_size)])
        self.normal_transform.extend([transforms.ToTensor(), normalize])
        self.normal_transform = transforms.Compose(self.normal_transform)
    def get_imgs(self,ims_root):
        images = os.listdir(ims_root)
        clusters = [int(im.split('/')[-1].split('_')[0]) for im in images]
        images=[os.path.join(ims_root, image) for image in images]
        return images,np.array(clusters)
    def ensure_3dim(self, img):
         if len(img.size)==2:
             img = img.convert('RGB')
         return img
    def __getitem__(self, index):
        img = self.ensure_3dim(Image.open(self.images_fn[index]))
        img = self.normal_transform(img)
        return img, self.clusters[index]
    def __len__(self):
        return len(self.images_fn)

class ImagesForTest(data.Dataset):

    def __init__(self, ims_root, file_list, imsize=224, bbxs=None, transform=None):


        self.root = ims_root
        self.images_fn, self.hash_code = self.get_imgs(self.root,file_list)
        self.imsize = imsize
        self.transform = transform
        self.f_norm = normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.normal_transform = []
        self.normal_transform.extend([transforms.Resize(256), transforms.CenterCrop(imsize)])
        self.normal_transform.extend([transforms.ToTensor(), normalize])
        self.normal_transform = transforms.Compose(self.normal_transform)
    def get_imgs(self,ims_root,file_list):

        with open(file_list) as f:
            lines = f.readlines()
        lines = lines[1:]
        files = [line.strip().split(',')[-1] for line in lines]
        hashs = [line.strip().split(',')[0] for line in lines]

        images=[os.path.join(ims_root, image) for image in files]

        return images, hashs


    def ensure_3dim(self, img):
         if len(img.size)==2:
             img = img.convert('RGB')
         return img


    def __getitem__(self, index):

        img = self.ensure_3dim(Image.open(self.images_fn[index]))
        img = self.normal_transform(img)

        return img, self.hash_code[index]

    def __len__(self):
        return len(self.images_fn)


class TuplesDataset(data.Dataset):

    def __init__(self,imgs_root,train_list,imsize,batch_p,batch_k,transform):
        self.imsize = imsize
        self.batch_p = batch_p
        self.batch_k = batch_k
        self.imgs_root = imgs_root
        self.dict = self.create_dict(self.imgs_root)
        self.keys = []
        self.f_norm = normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.normal_transform = []
        self.normal_transform.extend([transforms.RandomResizedCrop(size=imsize), transforms.RandomGrayscale(p=0.2),
                                       transforms.ColorJitter(0.2, 0.2, 0.2, 0.2), transforms.RandomHorizontalFlip(0.5)])
        self.normal_transform.extend([transforms.ToTensor(), normalize])
        self.normal_transform = transforms.Compose(self.normal_transform)
    def create_dict(self, ims_root):
        images = os.listdir(ims_root)
        self.iterations = len(images)//(self.batch_p*self.batch_k)
        clusters = [int(im.split('/')[-1].split('_')[0]) for im in images]
        images=[os.path.join(ims_root, image) for image in images]
        dict = {}
        for i in range(len(images)):
            if clusters[i] in dict:
                dict[clusters[i]]+=[images[i]]
            else:
                dict[clusters[i]] = [images[i]]
        self.cur = {}
        for key in dict.keys():
            random.shuffle(dict[key])
            self.cur[key] = 0
        return dict
    def create_tuple(self):
        self.batch_sample = []
        self.batch_label = []
        for i in range(self.iterations):
            sample_cls = np.random.choice(list(self.dict.keys()),self.batch_p,replace=False)
            imgs = []
            labels = [[cls]*self.batch_k for cls in sample_cls]
            for key in sample_cls:
                samples_img = np.random.choice(self.dict[key],self.batch_k,replace=False)
                imgs+=[samples_img]
            imgs = np.reshape(imgs,(-1))
            labels = np.reshape(labels,(-1))
            self.batch_sample += [imgs]
            self.batch_label += [labels]
    def ensure_3dim(self, img):
         if len(img.size)==2:
             img = img.convert('RGB')
         return img

    def __getitem__(self, index):

        output = []
        imgs = self.batch_sample[index]
        clss = self.batch_label[index]

        for path in imgs:
            img = self.ensure_3dim(Image.open(path))
            img = self.normal_transform(img)
            output+=[img.unsqueeze(0)]

        output = torch.cat(output,dim = 0)
        return output, clss
    def __len__(self):
        return self.iterations

class TuplesDataset_list(data.Dataset):

    def __init__(self,imgs_root, train_list,imsize,batch_p,batch_k,transform):
        self.imsize = imsize
        self.batch_p = batch_p
        self.batch_k = batch_k
        self.dict = self.create_dict(imgs_root, train_list)
        self.keys = []
        self.f_norm = normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.normal_transform = []
        self.normal_transform.extend([transforms.RandomResizedCrop(size=imsize), transforms.RandomGrayscale(p=0.2),
                                       transforms.ColorJitter(0.2, 0.2, 0.2, 0.2), transforms.RandomHorizontalFlip(0.5)])
        self.normal_transform.extend([transforms.ToTensor(), normalize])
        self.normal_transform = transforms.Compose(self.normal_transform)
    def create_dict(self, imgs_root, file_list):

        images, clusters = self.get_imgs(file_list, imgs_root)
        self.iterations = len(images)//(self.batch_p*self.batch_k)
        dict = {}
        for i in range(len(images)):
            if clusters[i] in dict:
                dict[clusters[i]]+=[images[i]]
            else:
                dict[clusters[i]] = [images[i]]
        return dict
    def create_tuple(self):
        self.batch_sample = []
        self.batch_label = []
        for i in range(self.iterations):
            sample_cls = np.random.choice(list(self.dict.keys()),self.batch_p,replace=False)
            imgs = []
            labels = [[cls]*self.batch_k for cls in sample_cls]
            for key in sample_cls:
                samples_img = np.random.choice(self.dict[key],self.batch_k,replace=False)
                imgs+=[samples_img]
            imgs = np.reshape(imgs,(-1))
            labels = np.reshape(labels,(-1))
            self.batch_sample += [imgs]
            self.batch_label += [labels]

    def ensure_3dim(self, img):
         if len(img.size)==2:
             img = img.convert('RGB')
         return img
    def __getitem__(self, index):

        output = []
        imgs = self.batch_sample[index]
        clss = self.batch_label[index]

        for path in imgs:
            img = self.ensure_3dim(Image.open(path))
            img = self.normal_transform(img)
            output+=[img.unsqueeze(0)]

        output = torch.cat(output,dim = 0)
        return output, clss
    def __len__(self):
        return self.iterations



    def get_imgs(self, train_list, img_root):
        with open(train_list) as f:
            lines = f.readlines()
        images = [item.strip().split()[0] for item in lines]
        images = [os.path.join(img_root,item) for item in images]
        clusters = [int(item.strip().split()[1]) for item in lines]
        return images,np.array(clusters)


if __name__=='__main__':
    img_root = '/workdir/lizhuo/dataset/mt20000/images/MTCV_merge_to_dishname_20191108_stage2_sku_2w_imgs_320_changemode_train_changemode'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),normalize
    ])
    train_dataset = TuplesDataset(img_root,224, 100,2,transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    train_loader.dataset.create_tuple()
    for step, (x, y) in enumerate(train_loader):
        pdb.set_trace()
