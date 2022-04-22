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


class Harris:

    def __init__(self, frame):

        self.R = self.__interestPoints__(frame)
        self.thresh = self.thresholdedR(self.R)
        self.suppressed = self.nonMaximalSupress(self.thresh)
        
        
    def gaussDeriv2D(self,sigma):
        x = np.array(range(0, 6*ceil(sigma)))
        y = np.transpose(x)
        Gx = ((x-3*ceil(sigma))/(2*pi*sigma**4)) * \
            (np.exp(-(((x-3*ceil(sigma))**2+(y-3*ceil(sigma))**2)/(2*sigma**2))))
        Gy = ((y-3*ceil(sigma))/(2*pi*sigma**4)) * \
            (np.exp(-(((x-3*ceil(sigma))**2+(y-3*ceil(sigma))**2)/(2*sigma**2))))
        return [Gx, Gy[np.newaxis]]

  

    def __interestPoints__(self, frame):
        alpha = 0.05
 
        temp = np.pad(frame, ((0, np.max(frame.shape) - np.min(frame.shape)), (0, 0)), 'constant')
        temp = cv2.GaussianBlur(temp, (3, 3), 10)
     
        Gx,Gy = self.gaussDeriv2D(10)
        Gx = cv2.filter2D(temp.astype(np.float32), -1, Gx)
        Gy = cv2.filter2D(temp.astype(np.float32), -1, Gy)
        print("here")
        GxGy = Gx * Gy
        Gx2 = Gx ** 2
        Gy2 = Gy ** 2
      
        Gaussianblurredx = cv2.GaussianBlur(Gx2, (3, 3), 1)
        Gaussianblurredy = cv2.GaussianBlur(Gy2, (3, 3), 1)
        Gaussianblurredxy = cv2.GaussianBlur(GxGy, (3, 3), 1)
        addedm = Gaussianblurredx+Gaussianblurredy
        R = Gaussianblurredx*Gaussianblurredy - np.square(Gaussianblurredxy) - alpha*np.square(Gaussianblurredx+Gaussianblurredy)
        R = R*-1

       
        
       
        
        return R
    
    def thresholdedR(self,R):
        R = R + np.abs(np.min(R))
        threshold = np.where(R <1e-15, 0, R)
        return threshold
    
    def nonMaximalSupress(self,threshold):
        w_size = 100
        suppress = []
        im_h, im_w = threshold.shape[:2]

        for r in np.arange(im_h - w_size + 1, step=w_size):
            for c in np.arange(im_w - w_size + 1, step=w_size):
                region = threshold[r:r+w_size, c:c+w_size]
                if np.unique(region).size > 0:
                    if np.max(np.unique(region)) > 0:
                        max_r, max_c = np.unravel_index(np.argmax(np.unique(region)), region.shape)
                        suppress.append((c+max_c, r+max_r))
                        
        return suppress

imgGray = color.rgb2gray(cv2.imread('C:\\Users\\Jaiydev Gupta\\Documents\\5524 project\\cse5524-project\\data\\angle_left.png')) # please be midful of where you are getting the image from
#imgGray = color.rgb2gray(img)
print(imgGray.shape)
interestpoints = Harris(imgGray)

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
plt.imshow(interestpoints.R)
plt.show()
# plt.gca().set_title('raw R', c='r')

# plt.subplot(1, 2, 2)
thresh = interestpoints.thresh
plt.imshow(thresh)


# plt.gca().set_title('thresholded', c='r')

# plt.suptitle('Harris')
plt.show()

plt.figure(figsize=(5, 5))
plt.gca().imshow(imgGray, cmap='gray')
plt.gca().scatter(*zip(*interestpoints.suppressed), s=1, c='r')
plt.title('suppressed overlay')
plt.show()