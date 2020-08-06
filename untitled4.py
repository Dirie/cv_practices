# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 15:30:22 2016

@author: dirie
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import linalg
import cv2


f1 = np.array([[-0.25,-0.5,0.0],[0.0,0.0,0.0],[0.0,0.5,0.25]])
f2 = np.array([[-0.25,-0.5,0.0],[0.0,0.0,0.0],[0.0,0.5,0.25]])
f3 = np.array([[-0.25,-0.5,0.0],[0.0,0.0,0.0],[0.0,0.5,0.25]])
f4 = np.array([[-0.25,-0.5,0.0],[0.0,0.0,0.0],[0.0,0.5,0.25]])

f1p = np.array([[1.25,-0.25,-4.0,-3.2,-0.5,-0.25,0.0],
                [1.5,0.25,-4.25,-3.75,0,0.75,0],
                [0,0,-4.25,-3.75,0.25,0.5,1.5],
                [-0.5,0,-3.25,-3,3,-0.5,1.5],
                [-10.5,-10.75,-10.25,-3.25,-0.5,-0.25,-1],
                [-10.25,-10,-9.75,-2,75,-1,-0.5,-1],
                [-0.5,0.25,1.25,1,0,0,0]])


print(f1p.shape,f1.shape)

p,rem = signal.deconvolve(f1p,f1) 

print(p)