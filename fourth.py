# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 14:18:39 2016

@author: vmuser
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import linalg

RGB = plt.imread("/home/vmuser/Downloads/kodak/IMAGE-0.bmp")
I = np.copy(RGB[:, :, 0])
plt.imshow(I,cmap = plt.cm.gray)
F = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])

G = signal.convolve2d(I, F, mode='same', boundary='fill')
#plt.imshow(G, cmap= plt.cm.gray)
plt.subplot(131)
#plt.imshow(G, cmap= plt.cm.gray)
plt.axis("off")
plt.subplot(132)
#plt.imshow(np.abs(G), cmap= plt.cm.gray)
plt.axis("off")
plt.subplot(133)
#plt.imshow(I, cmap= plt.cm.gray)
plt.axis("off")


Q = signal.convolve2d(I, F.T, mode='same', boundary='fill')
#plt.imshow(G, cmap= plt.cm.gray)
plt.subplot(131)
#plt.imshow(Q, cmap= plt.cm.gray)
plt.axis("off")
plt.subplot(132)
#plt.imshow(np.abs(Q), cmap= plt.cm.gray)
plt.axis("off")
plt.subplot(133)
#plt.imshow(I, cmap= plt.cm.gray)
plt.axis("off")

A = np.array([[0,1,0],[-1,0,1],[0,-1,0]])
print A

H = signal.convolve2d(I, A, mode='same', boundary='fill')
#plt.imshow(G, cmap= plt.cm.gray)
plt.subplot(131)
#plt.imshow(H, cmap= plt.cm.gray)
plt.axis("off")
plt.subplot(132)
#plt.imshow(np.abs(H), cmap= plt.cm.gray)
plt.axis("off")
plt.subplot(133)
plt.imshow(I, cmap= plt.cm.gray)
plt.axis("off")

B = np.array([[0,1,0.5,0.25,0.125],
              [-1,0,1,0.5,0.25],
              [-0.5,-1,0,1,0.5],
              [-0.25,-0.5,-1,0,1],
              [-0.125,-0.25,-0.5,-1,0]])
print B

V = signal.convolve2d(I, B, mode='same', boundary='fill')
#plt.imshow(G, cmap= plt.cm.gray)
plt.subplot(131)
#plt.imshow(V, cmap= plt.cm.gray)
plt.axis("off")
plt.subplot(132)
#plt.imshow(np.abs(V), cmap= plt.cm.gray)
plt.axis("off")
plt.subplot(133)
#plt.imshow(I, cmap= plt.cm.gray)
plt.axis("off")

dt = 0.1
x= np.arange(-3.0,3.0+dt , dt)[:, np.newaxis]
y= np.arange(-3.0,3.0+dt , dt)[np.newaxis,: ]
std = 1.0
F = np.exp(-(x**2 + y**2) / (2.0 * std))
plt.figure()
#plt.imshow(F, cmap = plt.cm.gray)

dt = 0.1
x= np.arange(-3.0,3.0+dt , dt)[:, np.newaxis]
y= np.arange(-3.0,3.0+dt , dt)[np.newaxis,: ]
#modify std to get the degree of bluriness
std = 0.5
F = np.exp(-(x**2 + y**2) / (2.0 * std))
plt.figure()
#plt.imshow(F, cmap = plt.cm.gray)

V = signal.convolve2d(I, F, mode='same', boundary='fill')
#plt.imshow(G, cmap= plt.cm.gray)
plt.subplot(121)
#plt.imshow(V, cmap= plt.cm.gray)
plt.axis("off")
plt.subplot(122)
#plt.imshow(I, cmap= plt.cm.gray)
plt.axis("off")

c = 1.0 / np.sum(F)
Fx = (-2.0 * c / std) * x * np.exp(-(x**2 + y**2) / (2.0 * std))
plt.figure()
plt.imshow(Fx, cmap = plt.cm.gray)
plt.title(r'$_{x}')

c = 1.0 / np.sum(F)
Fy = (-2.0 * c / std) * y * np.exp(-(x**2 + y**2) / (2.0 * std))
plt.figure()
plt.imshow(Fy, cmap = plt.cm.gray)
plt.title(r'$_{y}')


Z = signal.convolve2d(I, Fx, mode='same', boundary='fill')
#plt.imshow(G, cmap= plt.cm.gray)
plt.subplot(121)
plt.imshow(Z, cmap= plt.cm.gray)
plt.axis("off")
plt.subplot(122)
plt.imshow(I, cmap= plt.cm.gray)
plt.title("xfdsdfxsf")
plt.axis("off")

X = signal.convolve2d(I, Fy, mode='same', boundary='fill')
#plt.imshow(G, cmap= plt.cm.gray)
plt.subplot(121)
plt.imshow(X, cmap= plt.cm.gray)
plt.axis("off")
plt.subplot(122)
plt.imshow(I, cmap= plt.cm.gray)
plt.title("xfdsdfxsf")
plt.axis("off")

