import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import transform,data
from PIL import Image

lower = np.array([0,101,68])  #填入代码1的low值
upper = np.array([21,251,255]) #填入代码1的high值
t = 0

path1 = '地址1' # 地址1 图片集的地址
path2 = '地址2' # 地址2 图片集所在的地址

file_dir = path1
os.mkdir(path2+'\\p') #生成新文夹p 
for file in os.listdir(file_dir):
    img = cv2.imread(file_dir + '\\' + file)
    t = t + 1
    frame = img 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imwrite(path2+'\\p\\'+str(t)+'.jpg',res) 

file_dir = path2+'\\p'
k = 0
a = 0
for file in os.listdir(file_dir):
    a = a + 1
    img = np.array(Image.open(file_dir + '\\' + file))
    t = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][0]==0 & img[i][j][2]==0 & img[i][j][1]==0:
                t = t + 1
    if t/(img.shape[0]*img.shape[1]) > 0.8:
        k = k + 1
        oldname=file_dir+ os.sep + file
        newname=file_dir+ os.sep +'a'+str(k)+'.jpg'
        os.rename(oldname,newname)
    print(a,k,t/(img.shape[0]*img.shape[1]))