import typing
from unicodedata import ucd_3_2_0
import numpy as np
import skvideo.io
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray
from cv2 import COLOR_RGB2GRAY
import scipy
import math


class Tracker:
    def __init__(self, input_video, threshold=0.05, region_size=9):
        self.input = input_video
        self.curr_frame = 0
        self.region_size = region_size
        mei = np.abs(rgb2gray(input_video[1]) - rgb2gray(input_video[0]))
        mei = mei > threshold
        mei = scipy.ndimage.binary_opening(mei)
        mei = scipy.ndimage.binary_closing(mei)
        self.feature_points = np.argwhere(mei)
        plt.imshow(input_video[0])
        plt.scatter(self.feature_points[:, 1], self.feature_points[:, 0])
        plt.show()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.curr_frame < len(self.input) - 2:
            self.curr_frame += 1
            curr = self.input[self.curr_frame]
            next = self.input[self.curr_frame + 1]
            flows = np.zeros((self.feature_points.shape[0], 4))
            for i, (r, c) in enumerate(self.feature_points):
                offset = math.floor(self.region_size/2)
                region0 = rgb2gray(
                    curr[r-offset:r+offset+1, c-offset:c+offset+1])
                region1 = rgb2gray(
                    next[r-offset:r+offset+1, c-offset:c+offset+1])
                if region0.size >= self.region_size ** 2 and region1.size >= self.region_size ** 2:
                    f = self.flow2(region0, region1)
                    if f is not None:
                        flows[i] = np.insert(f, 0, [r, c])
                        self.feature_points[i] = (r+f[0], c+f[1])
            return flows
        raise StopIteration()

    def flow(self, patch0, patch1):
        patch0 = rgb2gray(patch0)
        patch1 = rgb2gray(patch1)
        Gx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]]) / 8
        Gy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]]) / 8

        fx = cv2.filter2D(patch1.astype(np.float32), -1, Gx)
        fy = cv2.filter2D(patch1.astype(np.float32), -1, Gy)
        denom = np.sqrt(fx**2+fy**2)
        with np.errstate(divide='ignore', invalid='ignore'):
            mag = -(patch1-patch0) / denom
            u = fx/denom * mag
            v = fy/denom * mag
        return (u, v, mag)

    def flow2(self, patch0, patch1):
        # patch0 = rgb2gray(patch0)
        # patch1 = rgb2gray(patch1)
        Gx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]]) / 8
        Gy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]]) / 8

        fx = cv2.filter2D(patch1.astype(np.float32), -1,
                          Gx).reshape((patch1.size, 1))
        fy = cv2.filter2D(patch1.astype(np.float32), -1,
                          Gy).reshape((patch1.size, 1))
        ft = -(patch1-patch0).reshape((patch1.size, 1))
        a = np.hstack((fx, fy))
        try:
            return (np.linalg.inv(a.T @ a) @ a.T @ ft).reshape(2)
        except:
            return None
