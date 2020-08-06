#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 00:50:03 2016

@author: dirie
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import linalg
import cv2


class image_Procesing(object):
    """
    this class will be implemented all staff in chapter 13.
    """

    def __init__(self,I):
        """Return a new Truck object."""
        print("Welcome to the all implementation of image processing functions.");
        self.I = I

    def whitening(self):
        """Return the whitening of an image."""
        print("whitening method:");
        I = self.I
        m = np.mean(I)
        s = np.std(I)
        cols, rows = I.shape
        w = np.zeros((cols,rows),dtype=float)
        for r in np.arange(0,rows):
            for c in np.arange(0,cols):
                w[r:c] = ((I[r:c] -m)/s)
        return w

        
    def Histogram_equalization(self):
        
        """Return the whitening of an image."""
        print("Histogram equalization:");
        I = self.I
        cols, rows = I.shape
        h = np.zeros((cols,rows),dtype=float)
        for r in np.arange(0,rows):
            for c in np.arange(0,cols):
                h[r:c] = I[r:c] +1
        return h

    def Gabor_filter(self):
        """Return the Gabor filter operation of an image."""
        print("whitening method:");
        m = np.linspace(-5,5,50)[:,np.newaxis]
        n = np.linspace(-5,5,50)[np.newaxis,:]
        s = 1.0
        freq = 0.5
        w = 2 * np.pi * freq
        lam = 100
        phi = np.pi/2
        fg = 1/(2 * np.pi * s) * np.exp(-(m**2 + n**2) / (2.0 * s))
        
        fw = np.sin(2 * np.pi * (np.cos(w) * m + np.sin(w) * n)/lam + phi)
        f =fg * fw
        G = signal.convolve2d(self.I,f,boundary='fill')
        return G
        
    def Rober_Operation(self):
        """Return the Gabor filter operation of an image."""
        print("whitening method:");
        Rx = np.array([[-1,0,0],[0,1,0],[0,0,0]])
        Ry = np.array([[0,-1,0],[1,0,0],[0,0,0]])
        gx = signal.convolve2d(self.I,Rx,boundary='fill')
        gy = signal.convolve2d(self.I,Ry,boundary='fill')
        maxp =0
        th = 0.1
        Rm = np.sqrt(gx**2 + gy**2)
        rows,cols = self.I.shape
        for r in np.arange(0,rows):
            for c in np.arange(0,cols):
                if Rm[r][c] >= maxp:
                    maxp = Rm[r][c]
        for r in np.arange(0,rows):
            for c in np.arange(0,cols):
                if Rm[r][c] >= th * maxp:
                    Rm[r][c] = 1
                else:
                    Rm[r][c] = 0
        return Rm
    def Perwit_Operation(self):
        """Return the Gabor filter operation of an image."""
        print("whitening method:");
        Px = np.array([[-0.333,-0.333,-0.333],[0,0,0],[0.333,0.333,0.333]])
        Py = np.array([[-0.333,0,0.333],[-0.333,0,0.333],[-0.333,0,0.333]])
        
        gx = signal.convolve2d(self.I,Px,boundary='fill')
        gy = signal.convolve2d(self.I,Py,boundary='fill')
        maxp =0
        th = 0.1
        Pm = np.sqrt(gx**2 + gy**2)
        rows,cols = self.I.shape
        for r in np.arange(0,rows):
            for c in np.arange(0,cols):
                if Pm[r][c] >= maxp:
                    maxp = Pm[r][c]
        for r in np.arange(0,rows):
            for c in np.arange(0,cols):
                if Pm[r][c] >= th * maxp:
                    Pm[r][c] = 1
                else:
                    Pm[r][c] = 0
        return Pm
        
    def Soble_Operation(self):
        """Return the Gabor filter operation of an image."""
        print("whitening method:");
        Sx = np.array([[-0.25,-0.5,-0.25],[0,0,0],[0.25,0.5,0.25]])
        Sy = np.array([[-0.25,0,0.25],[-0.5,0,0.5],[-0.25,0,0.5]])
        gx = signal.convolve2d(self.I,Sx,boundary='fill')
        gy = signal.convolve2d(self.I,Sy,boundary='fill')
        maxp =0
        th = 0.1
        Sm = np.sqrt(gx**2 + gy**2)
        rows,cols = self.I.shape
        for r in np.arange(0,rows):
            for c in np.arange(0,cols):
                if Sm[r][c] >= maxp:
                    maxp = Sm[r][c]
        for r in np.arange(0,rows):
            for c in np.arange(0,cols):
                if Sm[r][c] >= th * maxp:
                    Sm[r][c] = 1
                else:
                    Sm[r][c] = 0
        return Sm  

    def ft_filter(self):
        a = np.array([[2,0,0,7,6,7,7],
                     [0,1,1,6,7,7,7],
                     [0,0,1,7,7,5,7],
                     [0,0,2,7,6,7,5],
                     [1,0,0,7,5,7,5],
                     [6,6,7,7,7,7,7],
                     [7,6,5,5,7,7,7]])
        f = np.array([[-0.25,-0.25,0],
                      [-0.25,0,0.25],
                      [0,0.25,0.25],])
        C = signal.convolve2d(a,f,boundary='fill')
        print(C,C.shape)
        return C

I = plt.imread("lena.png")
P = image_Procesing(I)
#P.ft_filter()
plt.figure()
plt.subplot(111)
plt.imshow(I, cmap = plt.cm.gray)
plt.title("Original")

"""whitening function call """
#plt.figure()
#plt.subplot(111)
#plt.imshow(P.whitening(), cmap = plt.cm.gray)
#plt.title("Whitening")

"""histogram function call """
#plt.figure()
#plt.subplot(111)
#plt.imshow(P.Histogram_equalization(), cmap = plt.cm.gray)
#plt.title("histogram equalization")

"""histogram function call """
#plt.figure()
#plt.subplot(111)
#plt.imshow(P.Gabor_filter(), cmap = plt.cm.gray)
#plt.title("Gabor filter")

"""robert function call """
#plt.figure()
#plt.subplot(111)
#plt.imshow(P.Rober_Operation(), cmap = plt.cm.gray)
#plt.title("Robert operation")

"""perwit function call """
#plt.figure()
#plt.subplot(111)
#plt.imshow(P.Perwit_Operation(), cmap = plt.cm.gray)
#plt.title("perwit operation")

"""sobel function call """
plt.figure()
plt.subplot(111)
plt.imshow(P.Gabor_filter(), cmap = plt.cm.gray)
plt.title("Gabor filter")


