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


class Whiteboard:
    
    def __init__(self,video):
        
      
        
        self.Rstore = self.__interestPoints__(video)
        #self.Rstore = self.gaussDeriv(3)
        
    # def gaussDeriv(self,sigma):
    #     masksize = (2*ceil(3*sigma)+1);
    #     A1 = np.zeros((masksize, masksize))
        
    #     for j in range(masksize):
    #         rangestart = -3*sigma
    #         for k in range(masksize):
    #             A1[j][k] = rangestart
    #             rangestart+=1
                
    #     A2 = np.zeros((masksize,masksize))
    #     rangestart = -3*sigma
        
    #     for j in range(masksize):
            
    #         for k in range(masksize):
    #             A1[j][k] = rangestart
    #             rangestart+=1
    #         rangestart+=1
    #     calc = ((A1**2 + A2**2) * (1/(2*(3*sigma)^2)));
    #     eVal = np.e^calc
    #     return calc
                
        
        
        
    
    def __interestPoints__(self,videodata):
        
        arr = []
        #for x in range(len(videodata)): # for each image 
        temp = np.pad(videodata, ((0,233),(0,0)), 'constant') 
        temp = Gaussianblurredx = cv2.GaussianBlur(temp, (5,5), 1)
        # (plt.imshow( temp))
        # plt.show()
            #blur = cv2.GaussianBlur(temp,(3, 3), 3)
                    
            # image_first_derivative = laplace(temp) # first derivative of image 
            # image_first_derivative2 = laplace(temp) # second derivative of an image
        image_first_derivative = gaussian_filter(temp,sigma=1,order=[1,0],output=np.float64, mode='nearest') # first derivative of image 
        
       
        image_first_derivative2 = gaussian_filter(temp,sigma=1,order=[0,1],output=np.float64, mode='nearest')
        # image_first_derivative = laplace(videodata) # first derivative of image 
        # image_first_derivative2 = laplace(videodata)

        Gx2 = np.matmul(image_first_derivative,image_first_derivative);
        Gy2 = np.matmul(image_first_derivative2,image_first_derivative2);
        GxGy = np.matmul(image_first_derivative,image_first_derivative2);
        Gaussianblurredx = cv2.GaussianBlur(Gx2, (5, 5), cv2.BORDER_CONSTANT)
        Gaussianblurredy = cv2.GaussianBlur(Gy2, (5, 5), cv2.BORDER_CONSTANT)
        Gaussianblurredxy = cv2.GaussianBlur(GxGy, (5, 5),cv2.BORDER_CONSTANT)
        addedm = Gaussianblurredx+Gaussianblurredy
        R  = (Gaussianblurredx @ Gaussianblurredy) - (Gaussianblurredxy@Gaussianblurredxy) - (2*(addedm @ addedm))
        R = R*-1

        
        threshold = R > 10e7
        #threshold = R>7
        (plt.imshow(threshold))
        plt.show()
        exit
        arr.append(R)
            
        # return arr
        return R

# vid = np.squeeze(skvideo.utils.rgb2gray(skvideo.io.vread("C:\\Users\\Jaiydev Gupta\\Documents\\5524 project\\cse5524-project\\data\\up_move_right_Trim.mp4")));
img = io.imread('C:\\Users\\Jaiydev Gupta\\Documents\\5524 project\\cse5524-project\\data\\angle_left.png')
imgGray = cv2.imread('C:\\Users\\Jaiydev Gupta\\Documents\\5524 project\\cse5524-project\\data\\angle_left.png',0)
#imgGray = color.rgb2gray(img)
print(imgGray.shape)
whiteboard = Whiteboard(imgGray)
# (plt.imshow(whiteboard.Rstore))
# plt.show()
    
    
        
        
    
    


