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



# kindly just run the python file, all you need to do is change the path  of  the video down below 


vid = cv2.VideoCapture("C:\\Users\\Jaiydev Gupta\\Documents\\5524 project\\cse5524-project\\mspaint2.0\\whiteboard\\up_move_right_Trim_little.mp4") 

if (vid.isOpened()== False): 
  print("Error opening video stream or file")
count = 0

while(vid.isOpened()):
  count +=1
  # Capture frame-by-frame
  ret, frame = vid.read()
  if ret == True:
     
       
    
    temp = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(temp,(35, 35), 0)
    print()
    print()

    sobelx = cv2.Sobel(blur,cv2.CV_8U,1,0,ksize=5)

    thresh1 = sobelx >2;
    
#
    
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
    
            
    if count == 1 :          
        (plt.imshow( im_bw))
        plt.plot(loc[1], loc[0], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="green")
        plt.show()
        exit(1)
  

    
  

      



                
      
        
       
       
                
        
 