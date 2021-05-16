#coding:utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import time
from pylab import*


def NeumannBoundaryCond(f):
        [ny, nx] = f.shape
        g = f.copy()
    
        g[0, 0] = g[2, 2]
        g[0, nx-1] = g[2, nx-3]
        g[ny-1, 0] = g[ny-3, 2]
        g[ny-1, nx-1] = g[ny-3, nx-3]
    
        g[0, 1:-1] = g[2, 1:-1]
        g[ny-1, 1:-1] = g[ny-3, 1:-1]
    
        g[1:-1, 0] = g[1:-1, 2]
        g[1:-1, nx-1] = g[1:-1, nx-3]
    
        return g

def mat_math (intput,str):
    output = intput 
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if str == "atan":
                output[i,j] = math.atan(intput[i,j]) 
            if str == "sqrt":
                output[i,j] = math.sqrt(intput[i,j]) 
    return output 

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def shan_relu_2021(LSF, img, mu, nu, epison, step, lambda1, lambda2, kernel, numIter):
    for k1 in range(numIter):
        Drc = (epison / math.pi) / (epison*epison+ LSF*LSF)
        Hea = 0.5*(1 + (2 / math.pi)*mat_math(LSF/epison,"atan")) 
        LSF = NeumannBoundaryCond(LSF)
        Iy, Ix = np.gradient(LSF) 
        normDu = np.sqrt(Ix**2 + Iy**2 + eps)
        Nx = Ix / normDu 
        Ny = Iy / normDu
        Mxx,Nxx = np.gradient(Nx) 
        Nyy,Myy = np.gradient(Ny) 
        cur = Nxx + Nyy 
        Length = nu*Drc*cur 
    
        Lap = cv2.Laplacian(LSF,-1) 
        Penalty = mu*(Lap - cur) 
    
        KIH = cv2.filter2D(Hea*img,-1,kernel)
        KH = cv2.filter2D(Hea,-1,kernel)
        f1 = KIH / KH 
        KIH1 = cv2.filter2D((1-Hea)*img, -1,kernel)
        KH1 = cv2.filter2D(1-Hea, -1,kernel)
        f2 = KIH1 / KH1
        
        s1 = (f1 - img)
        s2 = (img - f2)
        
        FF1 = s1*lambda1*(cv2.filter2D((img - f1)**2, -1, kernel))
        FF2 = s2*lambda1*(cv2.filter2D((img - f2)**2, -1, kernel))
        dataForce = FF1 - FF2 
        A = Drc*dataForce
    
        LSF = LSF + step*(Length + Penalty + A) 
    #plt.imshow(s, cmap ='gray'),plt.show() 
    return LSF 


global eps
eps = 10e-10

Image = cv2.imread('f95.bmp',1)
image = cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
img = np.array(image,dtype=np.float64) 
img = img + eps


IniLSF = np.ones((img.shape[0],img.shape[1]),img.dtype) 
IniLSF[25:35, 25:45]= -1 
IniLSF = -IniLSF 

Image = cv2.cvtColor(Image,cv2.COLOR_BGR2RGB)
#Kernel
lambda1 = 10
lambda2 = 10
mu = .1
nu = 0.01*255*255 
num = 30
numIter = 1
epison = 4 
step = .1 
sig = 5
LSF = IniLSF
shape = (sig*4+1,sig*4+1)
kernel = matlab_style_gauss2D(shape = shape, sigma = sig)
#kernel = np.ones(shape) #/(sig*4+1)**2

plt.clf()
plt.figure(1)
plt.imshow(Image)
plt.xticks([])
plt.yticks([])
plt.contour(IniLSF,[0], colors = 'green',linewidth=2)
plt.draw()
plt.show(block=False) 



tic = time.time()

for i in range(1,num+1):
    LSF = shan_relu_2021(LSF, img, mu, nu, epison,step, lambda1, lambda2, kernel, numIter)
    if i % 2 == 0:
        plt.clf()
        plt.imshow(Image)
        plt.title(str(i) + ' th iteration')
        plt.xticks([])
        plt.yticks([])  
        plt.contour(LSF,[0], colors = 'red',linewidth=2) 
        plt.draw()
        plt.show(block=False)
        plt.pause(0.01)
        
plt.clf()
plt.imshow(Image)
plt.title(str(i) + ' th FInal iteration')
plt.xticks([])
plt.yticks([])  
plt.contour(LSF,[0], colors='red',linewidth=2) 
plt.contour(IniLSF,[0], colors = 'green',linewidth=2)
plt.draw()
plt.show(block=False)
plt.pause(0.01)

print('time_needed = ' + str(time.time() - tic))
