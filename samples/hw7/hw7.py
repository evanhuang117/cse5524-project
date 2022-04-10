# %%
from skimage.segmentation import slic
from skimage.measure import label,regionprops
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
from skimage.segmentation import mark_boundaries

# %% 1.


img = plt.imread('corgi-dog.jpg')
img2 = plt.imread('corgi-dog.jpg')
num_seg = 250
plt.figure()
plt.imshow(img)
segments_slic = slic(img, n_segments=num_seg, compactness=10, channel_axis=2)
plt.figure()
plt.imshow(mark_boundaries(img, segments_slic, color=(255, 0, 0)))
for seg_no in range(num_seg):
    seg_pxs = tuple(np.argwhere(segments_slic == seg_no).T.tolist())
    img[seg_pxs] = np.ma.average(img[seg_pxs], axis=0)
    
plt.imshow(img)

# %% 2.


def ncc(patch, tgt):
    ncc_val = np.zeros(3)
    for color in range(2):
        color_patch = patch[:, :, color]
        color_tgt = tgt[:, :, color]
        p_avg = np.average(color_patch)
        t_avg = np.average(color_tgt)
        p_std = np.std(color_patch)
        t_std = np.std(color_tgt)
        for r, c in np.ndindex(color_patch.shape):
            ncc_val[color] += (color_patch[r, c] - p_avg) * \
                (color_tgt[r, c] - t_avg) / p_std * t_std
    return np.sum(ncc_val / (patch.size - 1))


template = plt.imread('template.png')
search = plt.imread('search.png')

offset_r = math.floor(template.shape[0]/2)
offset_c = math.floor(template.shape[1]/2)
limits_r = offset_r, search.shape[0] - offset_r
limits_c = offset_c, search.shape[1] - offset_c
regions = [template[r-offset_r:r+offset_r, c-offset_c:c+offset_c, :]
           for r in range(*limits_r)
           for c in range(*limits_c)]

match_dist = np.zeros((search.shape[0]-template.shape[0],
                       search.shape[1]-template.shape[1]))
match_dist = np.zeros((search.shape[0], search.shape[1]))
i = 0
for r in range(*limits_r):
    for c in range(*limits_c):
        match_dist[r, c] = ncc(regions[i], template)
        i += 1

# %%
plt.imshow(match_dist)

# %%


def show_kth_matches(img, *args):
    plt.imshow(img)
    ax = plt.gca()
    ax.imshow(img)
    for i in args:
        i_min = np.unravel_index(np.argpartition(
            match_dist, i, axis=None)[i], match_dist.shape)
        # print(C[temp])
        # print(match_dist[i_min])
        # print(i_min)
        x = i_min[0]
        y = i_min[1]
        rect = patches.Rectangle(
            (x, y), 69, 47, linewidth=2, edgecolor='r', facecolor="none", zorder=100)
        ax.add_patch(rect)
    plt.show()


show_kth_matches(search, 1)

# %%
