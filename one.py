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

def histogram(I):
    h = np.zeros((256,))
    Rows,Cols = I.shape
    for r in np.arange(Rows):
        for c in np.arange(Cols):
            h[I[r,c]] = h[I[r,c]] + 1            
    return h
histogram(I)