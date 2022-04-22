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
from harris import Harris

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
        sig = 10
        k = 1.1
      
        log1 = ndimage.gaussian_laplace(frame, sigma=sig*k**0)
        
        log2 = ndimage.gaussian_laplace(frame, sigma=sig*k**1)
        
        log3 = ndimage.gaussian_laplace(frame, sigma=sig*k**2)

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
                    store = ((y,x,log2Max,sig*k**1))
                    tempo = 10*1.2**1
                    maxi = log2Max
                else:
                    store = ((y,x,log1Max,sig*k**0))
                    tempo = sig*k**0
                    maxi = log1Max
                    
                if log3Max >= maxi:
                    store = ((y,x,log3Max,sig*k**3))
                    maxi = log3Max
                else:
                    store = (y,x,maxi,tempo)
                
                if( maxi > 0.0 and maxi <1e-10) :
            
                    arr.append(store)

                
        # print(len(arr))        
        return arr
    
store2 = np.zeros((2017,2017))
imgGray = color.rgb2gray(cv2.imread('C:\\Users\\Jaiydev Gupta\\Documents\\5524 project\\cse5524-project\\mspaint2.0\\whiteboard\\angle_left.png')) # please be midful of where you are getting the image from
imgGray = np.pad(imgGray, ((0, np.max(imgGray.shape) - np.min(imgGray.shape)), (0, 0)), 'constant')
#imgGray = color.rgb2gray(img)
print(imgGray.shape)
interestpoints = Harris(imgGray)
suppress = interestpoints.suppressed
for x in range (len(interestpoints.suppressed)):
    store2[suppress[x][0]][suppress[x][1]] = imgGray[suppress[x][0]][suppress[x][1]]
    
        
laplaOfGauss = Log(store2)
store = laplaOfGauss.arr
print(store)
(plt.imshow( imgGray))
for x in store:
    plt.plot(x[1], x[0], marker="o", markersize=round(x[3]), markeredgecolor="red" )
plt.show()