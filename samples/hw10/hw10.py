# %%

from timeit import default_timer as timer
import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, floor,  pi, sqrt
import scipy.ndimage
from scipy.signal import fftconvolve


# %%

def ncc(patch, tgt):
    ncc_val = 0
    p_avg = np.average(patch)
    t_avg = np.average(tgt)
    p_std = np.std(patch)
    t_std = np.std(tgt)
    for r, c in np.ndindex(patch.shape):
        ncc_val += (patch[r, c] - p_avg) * \
            (tgt[r, c] - t_avg) / p_std * t_std
    return np.sum(ncc_val / (patch.size - 1))


def norm_data(data):
    mean_data = np.mean(data)
    std_data = np.std(data, ddof=1)
    return (data-mean_data)/(std_data)


def ncc2(patch, tgt):
    p_avg = np.mean(patch)
    t_avg = np.mean(tgt)
    p_std = np.std(patch)
    t_std = np.std(tgt)
    norm_p = (patch-p_avg)/(p_std)
    norm_t = (tgt-t_avg)/(t_std)
    return (1.0/(patch.size-1)) * np.sum(norm_p*norm_t)


# %%
right = plt.imread('right.png')
left = plt.imread('left.png')

disparity = np.zeros(right.shape)
start = timer()
for r in range(5, right.shape[0] - 5):
    for c in range(right.shape[1] -6, 5, -1):
        left_patch = left[r-5:r+6, c-5:c+6]
        ncc_vals = []
        max_c = -1
        max_ncc = -1
        for offset in range(50):
            offset_c = c - offset
            if offset_c >= 5:
                right_patch = right[r-5:r+6, offset_c-5:offset_c+6]
                # val = ncc2(right_patch, left_patch)
                # if val > max_ncc:
                #     max_ncc = val
                #     max_c = offset_c
                ncc_vals.append(ncc2(left_patch, right_patch))
        max_c = np.argmax(ncc_vals)
        disparity[r, c] = max_c

end = timer()
print(end - start)


# %%
filter = np.copy(disparity)
filter[filter >= 50] = 50
filter[filter <= 0] = 0
plt.figure()
plt.imshow(disparity, cmap='gray')
plt.figure()
plt.imshow(filter, cmap='gray')
# 
# from matlab API for the clims imagesc option thats used: "[cmin, cmax] - values <= cmin map to the first color in the cmap, values >= cmax map to the last color in the cmap, values between linearly map to the cmap"

# %%
