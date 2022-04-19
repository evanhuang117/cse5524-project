from operator import matmul
import skvideo.io  
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, laplace
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import ceil, floor,  pi, sqrt
import imageio as iio
from PIL import Image
from skimage import color
from skimage import io
from scipy import ndimage, misc

class Log:
    def __init__(self, frame):
        self.arr = self.sigmaMaximizer(frame)
         
    def findMax(self,log2, y ,x):
        point1 = log2[y][x]
        point2 = log2[y-1][x]
        point3 = log2[y+1][x]
        point4 = log2[y][x+1]
        point5 = log2[y][x-1]
        point6 = log2[y-1][x-1]
        point7= log2[y-1][x+1]
        point8 = log2[y+1][x+1]
        point9 = log2[y+1][x-1]
        a= [point1,point2,point3,point4,point5,point6,point7,point8,point9]
        return max(a)
    
    def sigmaMaximizer(self,frame):
        sigma = 10
        #  ndimage.gaussian_laplace(threshold, sigma=1)
        ## octave 1
        
        # level1 = cv2.GaussianBlur(threshold, (3,3), sigma)
        # level2 = cv2.GaussianBlur(level1, (3,3), sigma)
        # level3 = cv2.GaussianBlur(level2, (3,3), sigma)
        # level4 = cv2.GaussianBlur(level3, (3,3), sigma)
        log1 = ndimage.gaussian_laplace(frame, sigma=10*1.2**0)
        tup1 = (log1,sigma)
        log2 = ndimage.gaussian_laplace(frame, sigma=10*1.2**1)
        tup2 = (log2,sigma**2)
        log3 = ndimage.gaussian_laplace(frame, sigma=10*1.2**2)






        l,w = log2.shape    
        octave0 = []
        arr = []
        
        for y in range(1,len(log2)-1):
            for x in range(1, len(log2[0])-1):
                log2Max = self.findMax(log2,y,x)
                log1Max = self.findMax(log1,y,x)
                log3Max = self.findMax(log3,y,x)
                store = 0
                maxi = 0
                tempo = 0
                
                if log2Max >= log1Max:
                    store = ((y,x,log2Max,10*1.2**0))
                    tempo = 10*1.2**1
                    maxi = log2Max
                else:
                    store = ((y,x,log1Max,10*1.2**1))
                    tempo = 10*1.2**0
                    maxi = log1Max
                    
                if log3Max >= maxi:
                    store = ((y,x,log3Max,10*1.2**2))
                    maxi = log3Max
                else:
                    store = (y,x,maxi,tempo)
                
                    
                if(maxi>1e-10):
                
                    arr.append(store)
                # log2Max = max(log2Max,findMax(log1,y,x))
                # log2Max = max(log2Max,findMax(log3,y,x))
                
                
        return arr
    