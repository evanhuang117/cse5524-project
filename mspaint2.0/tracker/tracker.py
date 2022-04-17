import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import math


class Tracker:
    def __init__(self, input_video, threshold=0.05, region_size=9, rescale=0.5):
        self.input = input_video
        self.region_size = region_size
        self.rescale = rescale
        self.update_frames()
        self.feature_points = self.gen_feature_points(threshold)
        plt.imshow(self.curr_frame, cmap='gray')
        plt.scatter(self.feature_points[:, 1],
                    self.feature_points[:, 0], s=1, c='r')
        plt.show()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.update_frames():
            curr = self.curr_frame
            next = self.next_frame

            # 4d vectors : r, c, u, v
            # flows = self.feature_points.copy()
            # flows = np.hstack((flows, np.zeros((self.feature_points.shape[0], 2))))
            flows = np.zeros((self.feature_points.shape[0], 4))
            # go through all feature points we identified and calculate klt for each
            for i, (r, c) in enumerate(self.feature_points):
                # find region centered around feature point
                offset = math.floor(self.region_size/2)
                region0 = curr[r-offset:r+offset+1, c-offset:c+offset+1]
                region1 = next[r-offset:r+offset+1, c-offset:c+offset+1]
                # filter out border of region_size/2 around the image
                if region0.size >= self.region_size ** 2 and region1.size >= self.region_size ** 2:
                    flow_res = self.flow2(region0, region1)
                    # flow is none if we can't invert a mat
                    if flow_res:
                        f, mag = flow_res
                        # add the flow vector to the prev r,c to find the updated pos.
                        new_point = (int(np.ceil(r+f[0])),
                                     int(np.ceil(c+f[1])))
                        if new_point[0] < curr.shape[0] and new_point[0] >= 0 \
                                and new_point[1] < curr.shape[1] and new_point[1] >= 0:
                            # add updated r, c and flow vector to array of all flows
                            flows[i] = np.array([r, c, f[0], f[1]])
                            self.feature_points[i] = new_point
            return flows
        raise StopIteration()

    def update_frames(self):
        """
            using the input BGR video from cv2, updates the current and next
            frame to the grayscale frames, scaled according to self.rescale
            returns false when the last frame has been read
        """
        # for some reason uint8 breaks everything so need to divide
        _, self.curr_frame = self.input.read()
        self.curr_frame = cv2.cvtColor(
            self.curr_frame, cv2.COLOR_BGR2GRAY) / 255
         # rescale the image to make it easier to detect smaller movements
        width = int(self.curr_frame.shape[1] * self.rescale)
        height = int(self.curr_frame.shape[0] * self.rescale)
        dim = (width, height)

        self.curr_frame = cv2.resize(
            self.curr_frame, dim, interpolation=cv2.INTER_AREA)

        successful_read, self.next_frame = self.input.read()
        self.next_frame = cv2.cvtColor(
            self.next_frame, cv2.COLOR_BGR2GRAY) / 255
        self.next_frame = cv2.resize(
            self.next_frame, dim, interpolation=cv2.INTER_AREA)
        return successful_read


    def gen_feature_points(self, threshold):
        """
        find features to track based off first 2 frames using mei
        filters out the points so that there's only one per region of size region_size
        """
        mei = np.abs(self.next_frame - self.curr_frame)
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
        try:  # in case the matrix isn't invertible
            return (np.linalg.inv(a.T @ a) @ a.T @ ft).reshape(2), ft
        except:
            return None