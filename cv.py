# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 15:11:45 2016

@author: vmuser
"""

import matplotlib.pyplot as plt
from scipy import linalg
from scipy import signal
import numpy as np

RGB = plt.imread("/home/vmuser/Desktop/data/IMAGE-0.bmp")


def Gabor_Filter(sigma,phi,w,l,s=5):
    
    c =  1/(2*(np.pi)*sigma**2)
    GB = np.zeros((6*s,6*s),float)
    i = 0;
    for m in range(-3*s,3*s):
        for n in range(-3*s,3*s):
            #print i
            e = np.exp(-1*(m*m + n*n)/(2.0*sigma*sigma))
            b = np.sin((2*np.pi*(np.cos(w)*m+np.sin(w)*n)/(l+phi)))
            GB[m+3*s][n+3*s] = e*b
            i =i+1
    return GB

g = Gabor_Filter(0.8,np.pi / 2,2*np.pi,3)

m = np.linspace(-5, 5,30)[:,np.newaxis]
n = np.linspace(-5, 5,30)[np.newaxis,:]
s = 0.1
freq = 1
w = (np.pi/2) * freq
lam = 4
phi = np.pi
fg = 1.0/(2 * np.pi*s)*np.exp(- (m**2+n**2)/(2.0*s))
plt.close("all")
plt.figure(1)
plt.subplot(221),plt.imshow(g,cmap=plt.cm.gray),plt.title("Mein")
plt.subplot(222),plt.imshow(fg,cmap=plt.cm.gray),plt.title("Gaussian")
fw = np.sin(2*np.pi*(np.cos(w)*m + np.sin(w)*n)/lam+phi)
plt.subplot(223),plt.imshow(fw, cmap=plt.cm.gray),plt.title("Plane Wave")

f = fg*fw
plt.subplot(224),plt.imshow(f,cmap=plt.cm.gray),plt.title("Gabor") 
I = RGB[:,:,1]
f2 = signal.convolve2d(g,f, mode = "same", boundary = "fill")
I_new = signal.convolve2d(I,f, mode = "same", boundary = "fill")



#plt.figure(2)
#plt.imshow(I_new,cmap=plt.cm.gray),plt.title("Res") 
v = 1
ns = 30
x = np.linspace(-5*np.sqrt(v), 5*np.sqrt(v),ns)[:,np.newaxis]
y = np.linspace(-5*np.sqrt(v), 5*np.sqrt(v),ns)[np.newaxis,:]

Hx = (-m)/(v)*np.exp(-(n**2+m**2)/(2*v))
Hy = (-n)/(v)*np.exp(-(n**2+m**2)/(2*v))
Hx = Hx - np.average(Hx)
Hx = Hx/np.sum(np.abs(Hx))
Hy = Hy - np.average(Hy)
Hy = Hy/np.sum(np.abs(Hy))
Ix = signal.convolve2d(I,Hx, mode = "same", boundary = "fill")
Iy = signal.convolve2d(I,Hy, mode = "same", boundary = "fill")
G = np.sqrt(Ix**2+Iy**2)
theta = np.abs(np.arctan2(Iy,Ix))
#(Ix/Iy)

plt.figure(2)
plt.subplot(221),plt.imshow(Hx, cmap=plt.cm.gray),plt.title("$H_{x}$")
plt.subplot(222),plt.imshow(Hy, cmap=plt.cm.gray),plt.title("$H_{y}$")
plt.subplot(223),plt.imshow(G, cmap=plt.cm.gray),plt.title("$ G = \sqrt{I_{x}^{2}+I_{y}^{2}}$")

plt.subplot(224),plt.imshow(theta, cmap=plt.cm.gray),plt.title("$\Theta$")

plt.figure(3)
plt.subplot(221),plt.imshow(RGB),plt.title("normal") 
plt.subplot(222),plt.imshow(G,cmap=plt.cm.gray),plt.title("$ G = \sqrt{I_{x}^{2}+I_{y}^{2}}$")
plt.subplot(223),plt.imshow(Ix,cmap=plt.cm.gray),plt.title("Ix")
plt.subplot(224),plt.imshow(Iy,cmap=plt.cm.gray),plt.title("Iy")
plt.figure(4)
plt.hist(G.flatten(),bins = 40)

plt.figure(5)
plt.imshow(G>30,cmap=plt.cm.gray),plt.title("Iy")
theta_threshold = theta[G>np.avarage(G)]
#plt.figureplt
