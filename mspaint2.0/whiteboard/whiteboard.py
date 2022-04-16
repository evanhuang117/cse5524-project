from operator import matmul
import skvideo.io  
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, laplace
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import ceil, floor,  pi, sqrt



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
        for x in range(len(videodata)): # for each image 
            temp = np.pad(videodata[x], ((0,840),(0,0)), 'constant') 
            blur = cv2.GaussianBlur(temp,(3, 3), 3)
                    
            # image_first_derivative = laplace(temp) # first derivative of image 
            # image_first_derivative2 = laplace(temp) # second derivative of an image
            image_first_derivative = laplace(temp) # first derivative of image 
            image_first_derivative2 = laplace(temp)

            Gx2 = np.matmul(image_first_derivative,image_first_derivative);
            Gy2 = np.matmul(image_first_derivative2,image_first_derivative2);
            GxGy = np.matmul(image_first_derivative,image_first_derivative2);
            Gaussianblurredx = cv2.GaussianBlur(Gx2, (3, 3), 1)
            Gaussianblurredy = cv2.GaussianBlur(Gy2, (3, 3), 1)
            Gaussianblurredxy = cv2.GaussianBlur(GxGy, (3, 3), 1)
            addedm = Gaussianblurredxy+Gaussianblurredxy
            R  = np.matmul(Gaussianblurredx,Gaussianblurredy) - np.matmul(Gaussianblurredxy,Gaussianblurredxy) - (0.05*np.matmul(addedm,addedm)) 
            R = R*-1

            
            threshold = R > 2.5e6
            #threshold = R>0 
            (plt.imshow(threshold))
            plt.show()
            exit
            arr.append(R)
            
        return arr

vid = np.squeeze(skvideo.utils.rgb2gray(skvideo.io.vread("C:\\Users\\Jaiydev Gupta\\Documents\\5524 project\\cse5524-project\\data\\up_move_right_Trim.mp4")));
whiteboard = Whiteboard(vid)
(plt.imshow(whiteboard.Rstore[0]))
plt.show()
    
    
        
        
    
    


