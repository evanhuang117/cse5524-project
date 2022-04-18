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
        self.feature_points = self.gen_feature_points(threshold)
        plt.imshow(self.input[0], cmap='gray')
        plt.scatter(self.feature_points[:, 1],
                    self.feature_points[:, 0], s=1, c='r')
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
            # 4d vectors : r, c, u, v
            flows = np.zeros((self.feature_points.shape[0], 4))
            # go through all feature points we identified and calculate klt for each
            for i, (r, c) in enumerate(self.feature_points):
                # find region centered around feature point
                offset = math.floor(self.region_size/2)
                region0 = curr[r-offset:r+offset+1, c-offset:c+offset+1]
                region1 = next[r-offset:r+offset+1, c-offset:c+offset+1]
                # filter out border of region_size/2 around the image
                if region0.size >= self.region_size ** 2 and region1.size >= self.region_size ** 2:
                    f = self.flow2(region0, region1)
                    # flow is none if we can't invert a mat
                    if f is not None:
                        # add the flow vector to the prev r,c to find the updated pos.
                        new_point = (r+f[0], c+f[1])
                        if new_point[0] < curr.shape[0] and new_point[1] < curr.shape[1]:
                            # add updated r, c and flow vector to array of all flows
                            flows[i] = np.insert(f, 0, [r, c])
                            self.feature_points[i] = new_point
            return flows
        raise StopIteration()

    """
    find features to track based off first 2 frames using mei
    filters out the points so that there's only one per region of size region_size
    """
    def gen_feature_points(self, threshold):
            mei = np.abs(self.input[1] - self.input[0])
            mei = mei > threshold
            mei = scipy.ndimage.binary_opening(mei)
            mei = scipy.ndimage.binary_closing(mei)
            mei = np.argwhere(mei)
            print(mei.shape)
            mei_filtered = []
            for p in mei:
                # filter out points based on euclidean dist. between them
                if not mei_filtered or \
                        ((p[0]-mei_filtered[-1][0])**2 + (p[1]-mei_filtered[-1][1])**2)**.5 >= self.region_size:
                    mei_filtered.append(p)
            print(len(mei_filtered))
            return np.array(mei_filtered)

    def flow(self, patch0, patch1):
        patch0 = patch0
        patch1 = patch1
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
        try: # in case the matrix isn't invertible
            return (np.linalg.inv(a.T @ a) @ a.T @ ft).reshape(2)
        except:
            return None
