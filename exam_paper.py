# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 00:51:32 2016

@author: dirie
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import *
import cv2


class exam_paper(object):
    
    def __init__(self,im):
        print ('An implementatin started...')
        self.im = im
        
    def guassian_kernels(self,size,sizey=None):
        size = int(size)
        if not sizey:
            sizey = size
        else:
            sizey = int(sizey)
        y, x = mgrid[-size:size+1, -sizey:sizey+1]
        
        gx = -x * exp(-(x**2/float((0.5*size)**2) + y**2 /float((0.5*sizey)**2)))
        gy = -y * exp(-(x**2/float((0.5*size)**2) + y**2 /float((0.5*sizey)**2)))
        return gx,gy
    
    def gaussian(self,n,ny=None):
        gx , gy = self.guassian_kernels(n)
        
        imx = signal.convolve(self.im,gx, mode='same')
        imy = signal.convolve(self.im,gy, mode='same')
        
        return imx, imy
        
    def compute_harris_response(self):
        
        imx , imy = self.gaussian(3)
        gaus = cv2.getGaussianKernel(3,5)
        
        wxx = signal.convolve(imx * imx, gaus, mode='same')
        wxy = signal.convolve(imx * imy, gaus, mode='same')
        wyy = signal.convolve(imy * imy, gaus, mode='same')
        
        wdet = (wxx * wyy) - (wxy**2)
        wtr = wxx + wyy
        return wdet/wtr
        
    def get_haris_point(self,har, min_dis =10, th=0.1):
        print('this is for getting harris points')
        corner_thre = max(har.ravel()) * th
        har_t = (har> corner_thre) * 1
        
        cand = har_t.nonzero()
        coor = [(cand[0][c],cand[1][c]) for c in range(len(cand[0]))]
        cand_val = [har[c[0]][c[1]] for c in coor]
        
        ind = argsort(cand_val)
        
        allow_loc = zeros(har.shape)
        allow_loc[min_dis:-min_dis,min_dis:-min_dis] = 1
        
        filter_coor = []
        
        for i in ind:
            if allow_loc[coor[i][0]][coor[i][1]] == 1:
                filter_coor.append(coor[i])
                allow_loc[(coor[i][0]-min_dis):(coor[i][0]+min_dis),(coor[i][1]-min_dis):(coor[i][1]+min_dis)] = 0
        return filter_coor
        
        
    def showImage(self,pimg):
        plt.subplot(121),plt.imshow(self.im),plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(pimg,cmap = plt.cm.gray),plt.title('Image thresold')
        plt.xticks([]), plt.yticks([])
        plt.show()
        
        
        
I = plt.imread("test_image.png")[:,:,0]

P = exam_paper(I)
P.showImage(plt.imread("test_image.png")[:,:,1])
har = P.compute_harris_response()
fil_c = P.get_haris_point(har,6)

plt.figure()

plt.imshow(plt.imread("test_image.png")[:,:,1])
plt.plot([p[1] for p in fil_c], [p[0] for p in fil_c],'*')
plt.axis('off')
#plt.show()
plt.title("Harris conner detection")