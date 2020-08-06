# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal





def histogram(I):
    h = np.zeros((256,))
    g = np.arange(0,256)
#    Rows,Cols = I.shape
#    for r in np.arange(Rows):
#        for c in np.arange(Cols):
#            h[I[r,c]] = h[I[r,c]] + 1   

    for i in g:
        h[i] = np.sum(I == i)
    h = h / np.sum(h)
    return h,g
    
def histogram_equalization(I):
    E = np.copy(I)
    h,g = histogram(I)
    t = np.ones((256,))/256.0
    T = np.zeros((256,))
    T[0] = t[0]
    for i in np.arange(1,256):
        T[i] = T[i-1] + t[i]
    H = np.zeros((256,))
    H[0] = h[0]
    for i in np.arange(1,256):
        H[i] = H[i-1] + h[i]
    m = np.zeros((256,))
    for i in np.arange(0,256):
        j = np.argmin(np.abs(T-H[i]))
        m[i] = j
    for i in np.arange(0,256):
        E[E==i] = m[i]
    return E,m,g
    

I = plt.imread("/home/vmuser/Downloads/kodak/IMAGE-0.bmp")
#I_temp = plt.imread("/home/vmuser/Downloads/kodak/IMAGE-0.bmp")
#I = np.copy((I_temp * 255).astype(int))

#I = I * 255
I.astype(int)
plt.figure()
plt.imshow(I/255.0)

#plt.imshow(I)
#print I.shape
#red chanel
R= I[:, :, 0]
#green chanel
G= I[:, :, 1]
#blue chanel
B= I[:, :, 2]

plt.figure()
h,g = histogram(R)
plt.plot(g, h,color = 'red')
plt.title('R') 
h,g = histogram(G)
plt.plot(g, h,color = 'green')
plt.title('G')
h,g = histogram(B)
plt.plot(g, h,color = 'blue')
plt.title('B')

#plt.plot(h,g)
O1 , m ,g = histogram_equalization(R)
O2 , m ,g = histogram_equalization(G)
O3 , m ,g = histogram_equalization(G)
O = np.zeros(I.shape)
O[:,:,0] = O1
O[:,:,1] = O2
O[:,:,2] = O3

plt.figure()

plt.imshow(O/255.0)
