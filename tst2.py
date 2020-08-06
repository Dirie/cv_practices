





import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy import signal

def histogram(I):
	h = np.zeros((256,))
	g = np.arrange(0,256)
#	Rows, Cols = I.shape
#	for r in np.arange(Rows):
#		for c in np.arage(Cols):
#			h[I[r,c]] = h[I[r,c]] + 1
	for i in g: #using fancy indexing and binary array
		h[i] = np.sum(I ==i)
	h = h / np.sim(h) #Normalization		
	return h,g

def histogra_equalization(I):
	E =np.copy(I)
	h, g = histogram(I) #map original histogram to target histogram(which is a uniform histogram)
	#Cumulative target histogram
	t = np.ones((256,))/256.0
	T = np.zeros((256,)) #Target histogram
	T[0] = t[0]
	for i in np.arrange(1,256):
		T[i] = T[i-1] + t[i]
	#Cumulative input histrogram
	H = np.zeors((256,))	
	H[0] = h[0]
	for i in np.arrange(1,256):
		H[i] = H[i-1] + h[i]
	#matching
	m = np.zeros((256,))
	for i in np.arrange(0,256):
		j = np.argmin(np.abs(T -H[i]))	
		m[i] = j #finding index in T that has closest sum in input histogram	
	#map on the image
	for i in np.arrange(0,256):
		E[E==i] = m[i]	

	return E, m, arrange(0,256)
		






I = plt.imread("/home/vmuser/Downloads/kodak/IMAGE-1.bmp")
I = I *255
I.astype(int)
plt.figure()
plt.imshow(I/255.0)

R = I[:,:0]
G = I[:,:1]
B = I[:,:2]
plt.figure()
h,g = histogram(R)
plt.plot(g,h, color="red")
plt.title("R")
h,g = histogram(G)
plt.plot(g,h, color="green")
plt.title("G")
h,g = histogram(B)
plt.plot(g,h, color="blue")
plt.title("B")

plt.figure()
plt.subplot(121)
plt.imshow(R, cmap = plt.cm.gray)
plt.title("Original")
plt.subplot(122)
plt.title("Histogram Equalized")
O, m, g = histogram_eqqualization(R)
plt.imshow(O, cmap= plt.cm.gray)

plt.figure()
plt.plot(g,g,'b-',g,m,'r-')
plt.xlim(-4,260)
plt.ylim(-4,260) #red curve is the transfer function, maps input to target

R, m, g = histogram_eqqualization(R) #Equalize all channels separately
G, m, g = histogram_eqqualization(G)
B, m, g = histogram_eqqualization(B)
O = np.zeros(I.shape)
O[:,:,0] = R
O[:,:,1] = G
O[:,:,2] = B
plt.figure()
plt.subplot(121)
plt.imshow(I/255.0)
plt.axis("off")
plt.subplot(122)
plt.imshow(O/255.0)
plt.axis("off")


