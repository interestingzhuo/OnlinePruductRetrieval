import cv2
import os
img_root = '/workdir/lizhuo/dataset/mt20000/images'
file_list = '/workdir/lizhuo/dataset/mt20000/retrieval_dict/train_list.txt'
with open(file_list) as f:
    lines = f.readlines()

files = [line.split()[0] for line in lines]

for file in files:
    path = os.path.join(img_root,file)
    img = cv2.imread(path)
    try:
        img = cv2.resize(img,(224,224))
    except:
        print(path)
