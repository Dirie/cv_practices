# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import signal

I = plt.imread("./kodak/IMAGE-0.bmp")

R = I[:, :, 0]
G = I[:, :, 1]
B = I[:, :, 2]



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

Rm = np.copy(R).astype(float)
Rm = Rm - np.mean(Rm)
Rm = Rm / np.std(Rm)

Gm = np.copy(G).astype(float)
Gm = Gm - np.mean(Gm)
Gm = Gm / np.std(Gm)

Bm = np.copy(B).astype(float)
Bm = Bm - np.mean(Bm)
Bm = Bm / np.std(Bm)

plt.figure()
plt.subplot(221)
plt.imshow(I)
plt.axis("off")
plt.title("Input")

plt.subplot(222)
plt.imshow(Rm, cmap = plt.cm.gray)
plt.axis("off")
plt.title("Red Norm")

plt.subplot(223)
plt.imshow(Gm, cmap = plt.cm.gray)
plt.axis("off")
plt.title("Green Norm")

plt.subplot(224)
plt.imshow(Bm, cmap = plt.cm.gray)
plt.axis("off")
plt.title("Blue Norm")