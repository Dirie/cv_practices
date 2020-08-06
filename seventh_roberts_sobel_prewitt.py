# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 14:18:39 2016

@author: vmuser
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import linalg

I = plt.imread("/home/vmuser/Downloads/kodak/IMAGE-0.bmp")[:,:,1]
#I = plt.imread("/home/vmuser/Downloads/classic_edge_detectors_1.0/lena.png")
#I = np.copy(RGB[:, :, 1])
#plt.imshow(I,cmap = plt.cm.gray)

R_row, R_col = I.shape

##########################

prewitt1 = np.array([[-0.333,-0.333,-0.333],[0,0,0],[0.333,0.333,0.333]])

prewitt2 = np.array([[-0.333,0,0.333],[-0.333,0,0.333],[-0.333,0,0.333]])

sobel1 = np.array([[-0.25,-0.5,-0.25],[0,0,0],[0.25,0.5,0.25]])

sobel2 = np.array([[-0.25,0,0.25],[-0.5,0,0.5],[-0.25,0,0.5]])

roberts1 = np.array([[-1,0],[0,1]])

roberts2 = np.array([[0,-1],[1,0]])

########################

robertsG1 = signal.convolve2d(I, roberts1, mode='same', boundary='fill')

robertsG2 = signal.convolve2d(I, roberts2, mode='same', boundary='fill')

prewittG1 = signal.convolve2d(I, prewitt1, mode='same', boundary='fill')

prewittG2 = signal.convolve2d(I, prewitt2, mode='same', boundary='fill')

sobelG1 = signal.convolve2d(I, sobel1, mode='same', boundary='fill')

sobelG2 = signal.convolve2d(I, sobel2, mode='same', boundary='fill')


############################

prewittmaximum = np.sqrt(prewittG1**2 + prewittG2**2)

robertsmaximum = np.sqrt(robertsG1**2 + robertsG2**2)

sobelmaximum = np.sqrt(sobelG1**2 + sobelG2**2)

maxP = 0
maxR = 0
maxS = 0
###############################
for i in np.arange(0, R_row):
    for j in np.arange(0, R_col):
        if prewittmaximum[i][j] >= maxM:
            maxP = prewittmaximum[i][j]

for i in np.arange(0, R_row):
    for j in np.arange(0, R_col):
        if prewittmaximum[i][j] >= 0.1 * maxM:
            prewittmaximum[i][j] = 1
        else:    
            prewittmaximum[i][j] = 0
###############################            
for i in np.arange(0, R_row):
    for j in np.arange(0, R_col):
        if robertsmaximum[i][j] >= maxM:
            maxR = robertsmaximum[i][j]

for i in np.arange(0, R_row):
    for j in np.arange(0, R_col):
        if robertsmaximum[i][j] >= 0.1 * maxM:
            robertsmaximum[i][j] = 1
        else:    
            robertsmaximum[i][j] = 0
###############################  

for i in np.arange(0, R_row):
    for j in np.arange(0, R_col):
        if sobelmaximum[i][j] >= maxM:
            maxS = sobelmaximum[i][j]

for i in np.arange(0, R_row):
    for j in np.arange(0, R_col):
        if sobelmaximum[i][j] >= 0.1 * maxM:
            sobelmaximum[i][j] = 1
        else:    
            sobelmaximum[i][j] = 0

###############################            

plt.figure()
plt.subplot(221)
plt.imshow(I, cmap = plt.cm.gray)
plt.axis("off")
plt.title("Input")

plt.subplot(223)
plt.imshow(robertsmaximum, cmap = plt.cm.gray)
plt.axis("off")
plt.title("Roberts Operator")

plt.subplot(224)
plt.imshow(prewittmaximum,cmap = plt.cm.gray)
plt.axis("off")
plt.title("Prewitt Operator")

plt.subplot(222)
plt.imshow(sobelmaximum, cmap = plt.cm.gray)
plt.axis("off")
plt.title("Sobel Operator")





#
#G = signal.convolve2d(I, F, mode='same', boundary='fill')
##plt.imshow(G, cmap= plt.cm.gray)
#plt.subplot(131)
##plt.imshow(G, cmap= plt.cm.gray)
#plt.axis("off")
#plt.subplot(132)
##plt.imshow(np.abs(G), cmap= plt.cm.gray)
#plt.axis("off")
#plt.subplot(133)
##plt.imshow(I, cmap= plt.cm.gray)
#plt.axis("off")
#
#
#Q = signal.convolve2d(I, F.T, mode='same', boundary='fill')
##plt.imshow(G, cmap= plt.cm.gray)
#plt.subplot(131)
##plt.imshow(Q, cmap= plt.cm.gray)
#plt.axis("off")
#plt.subplot(132)
##plt.imshow(np.abs(Q), cmap= plt.cm.gray)
#plt.axis("off")
#plt.subplot(133)
##plt.imshow(I, cmap= plt.cm.gray)
#plt.axis("off")
#
#A = np.array([[0,1,0],[-1,0,1],[0,-1,0]])
#print A
#
#H = signal.convolve2d(I, A, mode='same', boundary='fill')
##plt.imshow(G, cmap= plt.cm.gray)
#plt.subplot(131)
##plt.imshow(H, cmap= plt.cm.gray)
#plt.axis("off")
#plt.subplot(132)
##plt.imshow(np.abs(H), cmap= plt.cm.gray)
#plt.axis("off")
#plt.subplot(133)
#plt.imshow(I, cmap= plt.cm.gray)
#plt.axis("off")
#
#B = np.array([[0,1,0.5,0.25,0.125],
#              [-1,0,1,0.5,0.25],
#              [-0.5,-1,0,1,0.5],
#              [-0.25,-0.5,-1,0,1],
#              [-0.125,-0.25,-0.5,-1,0]])
#print B
#
#V = signal.convolve2d(I, B, mode='same', boundary='fill')
##plt.imshow(G, cmap= plt.cm.gray)
#plt.subplot(131)
##plt.imshow(V, cmap= plt.cm.gray)
#plt.axis("off")
#plt.subplot(132)
##plt.imshow(np.abs(V), cmap= plt.cm.gray)
#plt.axis("off")
#plt.subplot(133)
##plt.imshow(I, cmap= plt.cm.gray)
#plt.axis("off")
#
#dt = 0.1
#x= np.arange(-3.0,3.0+dt , dt)[:, np.newaxis]
#y= np.arange(-3.0,3.0+dt , dt)[np.newaxis,: ]
#std = 1.0
#F = np.exp(-(x**2 + y**2) / (2.0 * std))
#plt.figure()
##plt.imshow(F, cmap = plt.cm.gray)
#
#dt = 0.1
#x= np.arange(-3.0,3.0+dt , dt)[:, np.newaxis]
#y= np.arange(-3.0,3.0+dt , dt)[np.newaxis,: ]
##modify std to get the degree of bluriness
#std = 0.5
#F = np.exp(-(x**2 + y**2) / (2.0 * std))
#plt.figure()
##plt.imshow(F, cmap = plt.cm.gray)
#
#V = signal.convolve2d(I, F, mode='same', boundary='fill')
##plt.imshow(G, cmap= plt.cm.gray)
#plt.subplot(121)
##plt.imshow(V, cmap= plt.cm.gray)
#plt.axis("off")
#plt.subplot(122)
##plt.imshow(I, cmap= plt.cm.gray)
#plt.axis("off")
#
#c = 1.0 / np.sum(F)
#Fx = (-2.0 * c / std) * x * np.exp(-(x**2 + y**2) / (2.0 * std))
#plt.figure()
#plt.imshow(Fx, cmap = plt.cm.gray)
#plt.title(r'$_{x}')
#
#c = 1.0 / np.sum(F)
#Fy = (-2.0 * c / std) * y * np.exp(-(x**2 + y**2) / (2.0 * std))
#plt.figure()
#plt.imshow(Fy, cmap = plt.cm.gray)
#plt.title(r'$_{y}')
#
#
#Z = signal.convolve2d(I, Fx, mode='same', boundary='fill')
##plt.imshow(G, cmap= plt.cm.gray)
#plt.subplot(121)
#plt.imshow(Z, cmap= plt.cm.gray)
#plt.axis("off")
#plt.subplot(122)
#plt.imshow(I, cmap= plt.cm.gray)
#plt.title("xfdsdfxsf")
#plt.axis("off")
#
#X = signal.convolve2d(I, Fy, mode='same', boundary='fill')
##plt.imshow(G, cmap= plt.cm.gray)
#plt.subplot(121)
#plt.imshow(X, cmap= plt.cm.gray)
#plt.axis("off")
#plt.subplot(122)
#plt.imshow(I, cmap= plt.cm.gray)
#plt.title("xfdsdfxsf")
#plt.axis("off")
#
