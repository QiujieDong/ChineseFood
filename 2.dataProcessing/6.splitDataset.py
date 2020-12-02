#!/usr/bin/env python
# coding: utf-8

# # Data Processing

# In[2]:


from __future__ import print_function
from __future__ import division

from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Dense
from keras.layers import Input
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import sys
import imghdr
import cv2
import math
import io


# In[8]:


#generating class.txt
def generateClass(dataset_path, class_savePath, filelist):

    with open(class_savePath,'w') as f:
        for file_name in filelist:
            f.write(file_name)
            f.write('\n')
            
def deletIrregular(image_path, image_list):
     
    imageTypes_need = ['.jpg', '.jpeg', '.png']
    
    delet_irregular_count = 0
    
    for image in image_list:
        src = os.path.join(os.path.abspath(image_path), image)
        image_type = os.path.splitext(src)[-1]

        if not imghdr.what(src):  
            os.remove(src)  # delet corrupted image
            delet_irregular_count += 1 
        elif image_type in imageTypes_need:
            imageSize = os.path.getsize(src) / 1e3  # most abnormal image's getsizeof will exceed 200
            if  imageSize > 500:
                os.remove(src)
                delet_irregular_count += 1 
            elif len(io.BytesIO(cv2.imread(src)).read()) == 0:
                os.remove(src)
                delet_irregular_count += 1 
            else:
                continue
        else:
            os.remove(src)  # delet non-image data 
            delet_irregular_count += 1
            
    return delet_irregular_count

# Perceptual Hash Algorithm -  dHash        
def dhash(image):
    # convert image to 8*8
    image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_CUBIC)
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dhash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                dhash_str = dhash_str + '1'
            else:
                dhash_str = dhash_str + '0'
    result = ''
    for i in range(0, 64, 4):
        result += ''.join('%x' % int(dhash_str[i: i + 4], 2))
        
    return result        

# calculate the difference between hash1 and hash2
def campHash(hash1, hash2):
    n = 0
    #  If the hash length is different, the comparison cannot be made, and -1 is returned.
    if len(hash1) != len(hash2):
        return -1
    # If the hash length is same, traversing hash1 ahd hash2 for comparison.
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n

def imageHash(image_path, image_list):
    images_hash_set = []
    
    for image_name in image_list:
        src = os.path.join(os.path.abspath(image_path), image_name)
        image = cv2.imread(src)
        image_hash = dhash(image)
        images_hash_set.append(image_hash)
        
    return images_hash_set

def deletDuplicate(image_path):
    image_list = os.listdir(image_path)
    image_list.sort()
    images_hash_set = imageHash(image_path, image_list)

    delet_duplicate_count = 0
    for i in range(len(image_list)):
        for j in range((i+1), len(image_list)):
            distance_hash = campHash(images_hash_set[i], images_hash_set[j])
            if distance_hash == 0:
                try:
                    image2_path = os.path.join(os.path.abspath(image_path), image_list[j])
                    os.remove(image2_path)
                    delet_duplicate_count += 1
                except:
                    continue
                
    return delet_duplicate_count

def ImageRename(image_path):
        image_list = os.listdir(image_path)
        image_list.sort() # if the filelist is not sorted, some file will be replaced when repeating rename result in 

        rename_count = 0

        for image_name in image_list:
            src = os.path.join(os.path.abspath(image_path), image_name)
            image_type = os.path.splitext(src)[-1]

            dst = os.path.join(os.path.abspath(image_path), str(rename_count).zfill(4) + image_type)
            os.rename(src, dst)
#             print ('converting %s to %s ...' % (src, dst))
            rename_count += 1
    
        return rename_count

def splitDataset(image_path, train_savePath, test_savePath, filename):

        image_list = os.listdir(image_path)
        
        for image_name in image_list:
            if image_name.startswith('.'):
                image_list.remove(image_name)

        image_size = len(image_list)
        train_size = math.ceil(image_size * 0.9)
        
        # If filename does not exist, it will be created automatically. 
        #'w' means to append data. The original data in the file will not be cleared.
        with open(train_savePath,'a') as train:
            for file_name in image_list[:train_size]:
#                train.write(filename + '/' + os.path.splitext(file_name)[0])
				train.write(filename + '/' + file_name)
                train.write('\n')
        with open(test_savePath,'a') as test:
            for file_name in image_list[train_size:]:
#                test.write(filename + '/' + os.path.splitext(file_name)[0])
				test.write(filename + '/' + file_name)
                test.write('\n')
        return train_size, (image_size - train_size)


# In[7]:


dataset_path = '/home/gpu/Project/ChineseFood/ChineseFood/images/'

# save files' paths
class_savePath = '/home/gpu/Project/ChineseFood/ChineseFood/meta/class.txt'
train_savePath = '/home/gpu/Project/ChineseFood/ChineseFood/meta/train.txt'
test_savePath = '/home/gpu/Project/ChineseFood/ChineseFood/meta/test.txt'

filelist = os.listdir(dataset_path)
for file_name in filelist:
    if file_name.startswith('.'):
        filelist.remove(file_name)
filelist.sort()
# Start processing images.

#print('Start generating class.txt work...')
#generateClass(dataset_path, class_savePath, filelist)
#print('The work of generating class.txt has been done!')

for filename in filelist:
    image_path = os.path.join(os.path.abspath(dataset_path), filename)
    image_list = os.listdir(image_path)
    image_num = len(image_list)
    
    print('Processing images in {}: Total {} images.'.format(filename, image_num))
    
#   print('Start deleting irrgular images work...')
#    delet_irregular_num = deletIrregular(image_path, image_list)
#    print('Delet irregular: ', delet_irregular_num)
    
#    print('Start deleting duplicate images work...')
#    delet_duplicate_num = deletDuplicate(image_path)
#    print('Delet duplicate: ', delet_duplicate_num)
    
#    print('Start renaming images work...')
#    rename_num = ImageRename(image_path)
#    print('Rename: ', rename_num)
    
    print('Start creat dataset TXT work...')
    train_num, test_num = splitDataset(image_path, train_savePath, test_savePath, filename)
    print('train_size:{}, test_size:{}'.format(train_num, test_num))

# split dataset
if not os.path.isdir('./ChineseFood/test') and not os.path.isdir('./ChineseFood/train'):

    def copytree(src, dst, symlinks = False, ignore = None):
        if not os.path.exists(dst):
            os.makedirs(dst)
            shutil.copystat(src, dst)# only copy the status information of src to dst
        lst = os.listdir(src)
        if ignore:
            excl = ignore(src, lst)
            lst = [x for x in lst if x not in excl]
        for item in lst:
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if symlinks and os.path.islink(s):
                if os.path.lexists(d):
                    os.remove(d)
                os.symlink(os.readlink(s), d)
                try:
                    st = os.lstat(s)
                    mode = stat.S_IMODE(st.st_mode)
                    os.lchmod(d, mode)
                except:
                    pass # lchmod not available
            elif os.path.isdir(s):
                copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)

    def generate_dir_file_map(path):
        dir_files = defaultdict(list)
        with open(path, 'r') as txt:
            files = [l.strip() for l in txt.readlines()]
            for f in files:
                dir_name, id = f.split('/')
#                 dir_files[dir_name].append(id + '.jpg')
                dir_files[dir_name].append(id)
#                print(dir_name, id)
        return dir_files

    train_dir_files = generate_dir_file_map('./ChineseFood/meta/train.txt')
    test_dir_files = generate_dir_file_map('./ChineseFood/meta/test.txt')


    def ignore_train(d, filenames):
#        print(d)
        subdir = d.split('/')[-1]
        to_ignore = train_dir_files[subdir]
        return to_ignore

    def ignore_test(d, filenames):
#        print(d)
        subdir = d.split('/')[-1]
        to_ignore = test_dir_files[subdir]
        return to_ignore

    copytree('./ChineseFood/images', './ChineseFood/test', ignore=ignore_train)
    copytree('./ChineseFood/images', './ChineseFood/train', ignore=ignore_test)
    
else:
    print('Train/Test files already copied into separate folders.')