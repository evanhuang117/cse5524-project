import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import scipy.ndimage
import math


def flow(patch0, patch1):
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


def scale_frame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def track_regions(curr_frame, next_frame, scale=0.5, regions=None):
    """
        run KLT tracking on a region in a scaled down version of the video then 
        upscale it to the original image size. input images are assumed to be CV2 (BGR)

        :param scale: amount that each frame is scaled down to run KLT tracking on, improves detection of small movements
        :param regions: array of tuples of (r, c, region_size) to track

    """
    # convert to grayscale if needed
    if len(curr_frame.shape) == 3:
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    if len(next_frame.shape) == 3:
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    curr = scale_frame(curr_frame)
    next = scale_frame(next_frame)
    new_pos = regions.copy()
    for i, (r, c, win_size) in enumerate(regions):
        offset = int(math.floor(win_size/2) + win_size / 2)
        region0 = curr[r-offset:r+offset+1, c-offset:c+offset+1]
        region1 = next[r-offset:r+offset+1, c-offset:c+offset+1]
        if region0.size >= win_size ** 2 and region1.size >= win_size ** 2:
            flow_res = flow(region0, region1)
            # flow is none if we can't invert a mat
            if flow_res:
                f, mag = flow_res
                f_upscale = f / scale
                # scale the r, c  up to the original size
                r_upscale = r / scale
                c_upscale = c / scale
                # add the flow vector to the prev r,c to find the updated pos.
                new_point = (int(np.ceil(r+f[0])),
                             int(np.ceil(c+f[1])))
                new_point_upscale = (int(np.ceil(r_upscale+f_upscale[0])),
                                     int(np.ceil(c_upscale+f_upscale[1])))
                if new_point[0] < curr.shape[0] and new_point[0] >= 0 \
                        and new_point[1] < curr.shape[1] and new_point[1] >= 0:
                    # update position for the region
                    new_pos[i] = (*new_point_upscale, win_size)
    return new_pos


class Tracker:
    def __init__(self, input_video, threshold=0.05, region_size=9, scale=0.5, regions=None):
        """
            :param input_video: the video to process, cv2 VideoCapture object
            :param threshold: threshold value for img diff interest points detection 
            :param region_size: region_size x region_size window in which KLT is run for each interest point
            :param scale: amount that each frame is scaled down to run KLT tracking on, improves detection of small movements
            :param regions: array of tuples of (r, c, region_size) - if this is not None, KLT tracking will be run on each region
        """
        self.input = input_video
        self.region_size = region_size
        self.scale = scale
        self.curr_frame = None
        self.regions = regions
        # read in the first 2 frames
        self.update_frames()
        if regions is None:
            self.feature_points = self.gen_feature_points(threshold)
        plt.imshow(self.curr_frame_scaled, cmap='gray')
        plt.scatter(self.feature_points[:, 1],
                    self.feature_points[:, 0], s=1, c='r')
        plt.show()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.update_frames():
            if self.regions is None:
                return self.track_points()
            else:
                return self.track_region()
        raise StopIteration()

    def track_points(self):
        """
            tracks interest points and the area around them (self.region_size)
            instead of an entire region
        """
        curr = self.curr_frame_scaled
        next = self.next_frame_scaled

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
                flow_res = self.flow(region0, region1)
                # flow is none if we can't invert a mat
                if flow_res:
                    f, mag = flow_res
                    f_upscale = f / self.scale
                    # scale the r, c  up to the original size
                    r_upscale = r / self.scale
                    c_upscale = c / self.scale
                    # add the flow vector to the prev r,c to find the updated pos.
                    new_point = (int(np.ceil(r+f[0])),
                                 int(np.ceil(c+f[1])))
                    new_point_upscale = (int(np.ceil(r_upscale+f_upscale[0])),
                                         int(np.ceil(c_upscale+f_upscale[1])))
                    # make sure new points are within bounds of image
                    if new_point[0] < curr.shape[0] and new_point[0] >= 0 \
                            and new_point[1] < curr.shape[1] and new_point[1] >= 0:
                        # add updated r, c and flow vector to array of all flows
                        flows[i] = np.array(
                            [r_upscale, c_upscale, f_upscale[0], f_upscale[1]])
                        self.feature_points[i] = new_point
        return flows

    def track_region(self):
        """
            run KLT tracking on a region in a scaled down version of the video then 
            upscale it to the original image size

        """
        curr = self.curr_frame_scaled
        next = self.next_frame_scaled
        flows = np.zeros((self.regions.shape[0], 4))
        for i, (r, c, win_size) in enumerate(self.regions):
            offset = int(math.floor(win_size/2) + win_size / 2)
            region0 = curr[r-offset:r+offset+1, c-offset:c+offset+1]
            region1 = next[r-offset:r+offset+1, c-offset:c+offset+1]
            if region0.size >= win_size ** 2 and region1.size >= win_size ** 2:
                flow_res = self.flow(region0, region1)
                # flow is none if we can't invert a mat
                if flow_res:
                    f, mag = flow_res
                    f_upscale = f / self.scale
                    # scale the r, c  up to the original size
                    r_upscale = r / self.scale
                    c_upscale = c / self.scale
                    # add the flow vector to the prev r,c to find the updated pos.
                    new_point = (int(np.ceil(r+f[0])),
                                 int(np.ceil(c+f[1])))
                    new_point_upscale = (int(np.ceil(r_upscale+f_upscale[0])),
                                         int(np.ceil(c_upscale+f_upscale[1])))
                    if new_point[0] < curr.shape[0] and new_point[0] >= 0 \
                            and new_point[1] < curr.shape[1] and new_point[1] >= 0:
                        # add current r, c and flow vector to array of all flows
                        # so flows are "old" pos with the corresponding flow from that point
                        flows[i] = np.array(
                            [r_upscale, c_upscale, f_upscale[0], f_upscale[1]])
                        # update pos of region
                        self.regions[i] = (*new_point, win_size)
            return flows

    def update_frames(self):
        """
            using the input BGR video from cv2, updates the current and next
            frame to the grayscale frames, scaled according to self.rescale
            returns false when the last frame has been read
        """
        if self.curr_frame is None:
            # for some reason uint8 breaks everything so need to divide
            _, self.curr_frame = self.input.read()
            self.curr_frame = cv2.cvtColor(
                self.curr_frame, cv2.COLOR_BGR2GRAY) / 255
            # rescale the image to make it easier to detect smaller movements
            width = int(self.curr_frame.shape[1] * self.scale)
            height = int(self.curr_frame.shape[0] * self.scale)
            dim = (width, height)
            self.curr_frame_scaled = cv2.resize(
                self.curr_frame, dim, interpolation=cv2.INTER_AREA)

            successful_read, self.next_frame = self.input.read()
            self.next_frame = cv2.cvtColor(
                self.next_frame, cv2.COLOR_BGR2GRAY) / 255
            self.next_frame_scaled = cv2.resize(
                self.next_frame, dim, interpolation=cv2.INTER_AREA)
        else:
            width = int(self.curr_frame.shape[1] * self.scale)
            height = int(self.curr_frame.shape[0] * self.scale)
            dim = (width, height)
            self.curr_frame = self.next_frame
            self.curr_frame_scaled = self.next_frame_scaled
            successful_read, self.next_frame = self.input.read()
            self.next_frame = cv2.cvtColor(
                self.next_frame, cv2.COLOR_BGR2GRAY) / 255
            self.next_frame_scaled = cv2.resize(
                self.next_frame, dim, interpolation=cv2.INTER_AREA)

        return successful_read

    def gen_feature_points(self, threshold):
        """
        find features to track based off first 2 frames using mei
        filters out the points so that there's only one per region of size region_size
        """
        mei = np.abs(self.next_frame_scaled - self.curr_frame_scaled)
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
