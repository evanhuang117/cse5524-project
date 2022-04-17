from cmath import atan
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
import math


#vid = np.squeeze(skvideo.utils.rgb2gray(skvideo.io.vread("C:\\Users\\Jaiydev Gupta\\Documents\\5524 project\\cse5524-project\\data\\up_move_right_Trim.mp4")));

vid = cv2.VideoCapture("C:\\Users\\Jaiydev Gupta\\Documents\\5524 project\\cse5524-project\\data\\up_move_right_Trim.mp4")
# Check if camera opened successfully
if (vid.isOpened()== False): 
  print("Error opening video stream or file")
count = 0
# Read until video is completed
while(vid.isOpened()):
  count +=1
  # Capture frame-by-frame
  ret, frame = vid.read()
  if ret == True:
       #temp = np.pad(frame, ((0,840),(0,0)), 'constant') 
       
    
    temp = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # plt.imshow(temp)
    # plt.show()
    # print(temp.shape)
    blur = cv2.GaussianBlur(temp,(35, 35), 0)
    print()
    print()
    #image_first_derivative = gaussian_filter(frame,sigma=1,order=[1,0],output=np.float64, mode='nearest') # first derivative of image
    sobelx = cv2.Sobel(blur,cv2.CV_8U,1,0,ksize=5)
#    abs_sobel64f = np.absolute(sobelx64f)
#    sobel_8u = np.uint8(abs_sobel64f)
    thresh1 = sobelx >2;
    
#    cv2.imshow('Frame',thresh1)
#    print(thresh1)
    
    (thresh, im_bw) = cv2.threshold(sobelx, 128, 255,  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    centery = 540
    centerx = 960
    arr = []
    im_bw = im_bw/255;
    
    
    for y in range(len(im_bw)):
        for x in range(len(im_bw[0])):
            if im_bw[y][x] > 0:
                angle = math.atan2(y - centery,x - centerx)
                if angle < 0:
                    angle = angle + (2*math.pi)
                arr.append((angle,(y,x)))
                
        arr.sort(key=lambda x:x[1])
        #print(arr)
        
    peak= 0
    for x in range(1,len(arr)-1):
        if peak < abs(arr[x-1][0]- arr[x+1][0]):
            peak =  abs(arr[x-1][0]- arr[x+1][0])
            loc = arr[x][1]
        peak = max(peak, abs(arr[x-1][0]- arr[x+1][0]))
    print(loc)
        
        
 
            
    if count == 6:            
        (plt.imshow( im_bw))
        plt.plot(loc[1], loc[0], marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green")
        plt.show()

    
  

      


# for x in range(len(vid)): # for each image 
#         temp = vid[x].astype("uint8")
#         temp = np.pad(vid[x], ((0,840),(0,0)), 'constant') 
#         blur = cv2.GaussianBlur(temp,(3, 3), 3)
#         image_first_derivative = gaussian_filter(temp,sigma=1,order=[1,0],output=np.float64, mode='nearest') # first derivative of image
#         image_first_derivative2 = gaussian_filter(temp,sigma=1,order=[0,1],output=np.float64, mode='nearest')
#         thresh1 = image_first_derivative >2;
#         thresh2 = image_first_derivative2 >2;
#         # magnitude=np.sqrt(thresh1**2+thresh2**2)
#         # print(thresh1)
#         # (plt.imshow( thresh1))
#         # plt.show()
#         # for y in range(len(thresh1)):
#         #     for x1 in range(len(thresh1[y])):
#         #         math,atan()
#         (thresh, im_bw) = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#         cv2.imshow('threshold',thresh)
                
      
        
       
       
                
        
 