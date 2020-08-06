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
#plt.imshow(I,cmap = plt.cm.gray)

#I = I * 255

variance = 1

ns = 21

I.astype(np.uint8)
#plt.imshow(I, cmap = plt.cm.gray)
x = np.linspace(-3*np.sqrt(v),3*np.sqrt(v), ns)[np.newaxis, :]
y = np.linspace(-3*np.sqrt(v),3*np.sqrt(v), ns)[:, np.newaxis]

H = np.exp(- (x**2 + y**2)/(2 * variance))

Hx = -x / v * H

Hy = -y / v * H

#normalization of the values
Hx = Hx - np.average(Hx)
Hx = Hx / np.sum(np.abs(Hx))
Hy = Hy - np.average(Hy)
Hy = Hy / np.sum(np.abs(Hy))
#normalization of the values

plt.figure()
plt.subplot(131)
plt.imshow(H, cmap = plt.cm.gray)
plt.title("$H$")
plt.subplot(132)
plt.imshow(Hx, cmap = plt.cm.gray)
plt.title("$H_{x}$")
plt.subplot(131)
plt.imshow(Hy, cmap = plt.cm.gray)
plt.title("$H_{y}$")

Ix = signal.convolve2d(I, Hx, mode='same')
Iy = signal.convolve2d(I, Hy, mode='same')

G = np.sqrt(Ix**2 + Iy**2)
plt.figure()
plt.subplot(141)
plt.imshow(I, cmap = plt.cm.gray)
plt.title("$I$")
plt.subplot(142)
plt.imshow(Ix, cmap = plt.cm.gray)
plt.title("$Ix$")
plt.subplot(143)
plt.imshow(Iy, cmap = plt.cm.gray)
plt.title("$Iy$")
plt.subplot(144)
plt.imshow(G, cmap = plt.cm.gray)
plt.title("$G$")
plt.close("all")
theta =np.abs(np.arctan2(Iy,Ix))
plt.hist(G.flatten())
plt.close("all")
print np.max(G.flatten())
print np.max(I)
plt.imshow(G>20)
plt.imshow(G>30, cmap= plt.cm.gray)
plt.imshow(G>np.median(G), cmap= plt.cm.gray)
plt.imshow(G>np.average(G), cmap= plt.cm.gray)
plt.figure()


#theta_threshold = theta[G>np.max(G)]
theta_threshold = theta[G>np.average(G)]
#x axis in radians
# we can interpret by looking at the angles in radians
plt.hist(theta_threshold, bins =36)