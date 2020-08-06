# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

I = plt.imread("/home/vmuser/Downloads/kodak/IMAGE-0.bmp")
#I_temp = plt.imread("/home/vmuser/Downloads/kodak/IMAGE-0.bmp")
#I = np.copy((I_temp * 255).astype(int))

plt.imshow(I)
print I.shape
#red chanel
R= I[:, :, 0]
#green chanel
G= I[:, :, 1]
#blue chanel
B= I[:, :, 2]

plt.close("all")
plt.figure()
plt.subplot(221)
plt.imshow(I)
plt.axis("off")
plt.title("Input")

plt.subplot(222)
plt.imshow(R, cmap = plt.cm.gray)
plt.axis("off")
plt.title("Red")

plt.subplot(223)
plt.imshow(G, cmap = plt.cm.gray)
plt.axis("off")
plt.title("Green")

plt.subplot(224)
plt.imshow(B, cmap = plt.cm.gray)
plt.axis("off")
plt.title("Blue")

plt.figure()
plt.imshow(R[300:320, 280:300], cmap = plt.cm.gray)

#Normalization#
Rm = np.copy(R)
Rm = Rm - np.average(Rm)
Rm = Rm / np.std(Rm) 

Gm = np.copy(G)
Gm = Gm - np.average(Gm)
Gm = Gm / np.std(Gm)

Bm = np.copy(B)
Bm = Bm - np.average(Bm)
Bm = Bm / np.std(Bm)
#Normalization#

plt.figure()
plt.subplot(221)
plt.imshow(I)
plt.axis("off")
plt.title("Input")

plt.subplot(222)
plt.imshow(Rm, cmap = plt.cm.gray)
plt.axis("off")
plt.title("Red normalized")

plt.subplot(223)
plt.imshow(Gm, cmap = plt.cm.gray)
plt.axis("off")
plt.title("Green normalized")

plt.subplot(224)
plt.imshow(Bm, cmap = plt.cm.gray)
plt.axis("off")
plt.title("Blue normalized")


R_row, R_col = R.shape
G_row, G_col = G.shape
B_row, B_col = B.shape

im_sum = 0

for r in np.arange(0, R_row):
    for c in np.arange(0, R_col):
        
