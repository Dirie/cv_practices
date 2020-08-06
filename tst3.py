






import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy import signal
RGB = plt.imread("/home/vmuser/Downloads/kodak/IMAGE-1.bmp")

I = np.copy(RGB[:,:,0])
plt.imshow(I, cmap = plt.cm.gray)
F = np.array([[-1, -1, -1], [0,0,0], [1,1,1]]) #filter template- > higtens horizontal regions

plt.close("all")
G = signal.convolve2d(I, F, mode = "same", boundary = "fill") #flips filter in the horizontal and vertical directions --> handles filter boundaries by using circular appron
plt.figure()
plt.subplot(121)
plt.imshow(G, cmap = plt.cm.gray)
plt.axis("off")
plt.subplot(122)
plt.imshow(np.abs(G), cmap = plt.cm.gray)
plt.axis("off")

G = signal.convolve2d(I, F.T, mode = "same", boundary = "fill") #flips filter in the horizontal and vertical directions --> handles filter boundaries by using circular appron
plt.figure() #F.T above ensure we detect vertical regions
plt.subplot(121)
plt.imshow(G, cmap = plt.cm.gray)
plt.axis("off")
plt.subplot(122)
plt.imshow(np.abs(G), cmap = plt.cm.gray)
plt.axis("off")

dt = 0.1 #dt = 0.5 has a faster responce and is more detailed.this changes the size of the filter
y = np.arrange(-3.0,3+dt, dt)[:, np.newaxis] #column vector
x = np.arrange(-3.0,3+dt, dt)[np.newsxis, :] #row vector

std = 1.0 #
F = np.exp(-(x**2+y**2)/(2.0*s))
plt.figure()
plt.imshow(F, cmap = plt.cm.gray)
plt.title(r'$\sigma = 1.0$')

std = 3.0
F = np.exp(-(x**2+y**2)/(2.0*s))
plt.figure()
plt.imshow(F, cmap = plt.cm.gray
plt.title(r'$\sigma = 3.0$')

std = 0.5 
F = np.exp(-(x**2+y**2)/(2.0*std))
plt.figure()
plt.imshow(F, cmap = plt.cm.gray)
plt.title(r'$\sigma^2 = 0.25$')

c = 1.0 / np.sum(F)
Fx = -c /(2.0 * s) * x * np.exp(-(x**2+y**2)/(2.0*std))
plt.figure()
plt.imshow(Fx, cmap = plt.cm.gray)
plt.title(r'$\F_[x]$') #light +ve numbers on the LHS dark  are -ve numbers on the RHS

Fy = -c /(2.0 * s) * y * np.exp(-(x**2+y**2)/(2.0*std))
y * np.exp(-(x**2+y**2)/(2.0*s))
plt.figure()
plt.imshow(Fy, cmap = plt.cm.gray)
plt.title(r'$\F_[y]$') #light +ve numbers on top dark  are -ve numbers on the bottom

Ix = sigmal.convolve2d(I, Fx, mode = "same", boundary = "fill")
plt.figure()
plt.imshow(np,.abs(Ix), cmap = plt.cm.gray)
plt.title(r'$\I_[x]$')

Iy = sigmal.convolve2d(I, Fy, mode = "same", boundary = "fill")
plt.figure()
plt.imshow(np,.abs(Iy), cmap = plt.cm.gray)
plt.title(r'$\I_[y]$')











#setting plot limits
plt.xlim(np.min(x), np.max(x))
plt.ylim(np.min(y), np.max(y))



F2 = np.array([[0, 1, 0.5, 0.25, 0,125], [-1, 0, 1, 0.5, 0.25], [-0.5, -1, 0, 1, 0.25], [-0.25, -0.5, -1, 0, 1], [-0.125, -0.25, -0.5, -1, 0]]) #45 degrees enhancement
plt.close("all")
G = signal.convolve2d(I, F2, mode = "same", boundary = "fill") #flips filter in the horizontal and vertical directions --> handles filter boundaries by using circularappron
plt.figure()
plt.subplot(121)
plt.imshow(G, cmap = plt.cm.gray)
plt.axis("off")
plt.subplot(122)
plt.imshow(np.abs(G), cmap = plt.cm.gray)
plt.axis("off")
