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
plt.imshow(I,cmap = plt.cm.gray)



m = np.linspace(-5, 5, 31)[:,np.newaxis]
n = np.linspace(-5, 5, 31)[np.newaxis,:]

s = 0.5
freq = 10
#w is the radial frequency
w = np.pi / 2
#w = 2 * np.pi * freq
#lambda is the wave lenght, when lambda changes, the distance between 2 peaks
#chages
lam = 2
#phi determines the shift of a wave
phi = 0
# check for phi 0, 45 and 90 degree
fg = 1.0 / (2 * np.pi * s) * np.exp(-(m**2 +n**2) / (2.0 * s))

plt.figure()
plt.subplot(131)
plt.imshow(fg, cmap = plt.cm.gray)
plt.title("Gaussian")
fw = np.sin(2 * np.pi * (np.cos(w) * m + np.sin(w)* n)/lam + phi)
plt.subplot(132)
plt.imshow(fw, cmap = plt.cm.gray)
plt.title("Plane wave")
f = fg * fw
f = f - np.average(f)
f = f / np.sum(np.abs(f))
plt.subplot(133)
plt.imshow(fw, cmap = plt.cm.gray)
plt.title("Gabor")
plt.imshow(f,cmap = plt.cm.gray)




#for phi in range(0,np.pi,0.1):
#    for w in range(0,1,0.1):
#        for lamba in range(1,100,10):
#            for s in range(0.5,5,5):
                
plt.figure()
V = signal.convolve2d(I, f, mode='same', boundary='fill')
plt.imshow(np.abs(V), cmap= plt.cm.gray)

np.max(np.abs(f))
plt.imshow(np.abs(f)>0.05)
np.max(np.abs(V))
plt.imshow(np.abs(V)>20)
plt.imshow(np.abs(V)>10,cmap = plt.cm.gray)



