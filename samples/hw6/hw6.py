# %%
import matplotlib.patches as patches
import matplotlib
import cv2
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from scipy import linalg


# %%


def cov(region, mu, r, c):
    indices = np.array(list(np.ndindex(region[:, :, 0].shape)))
    indices[:, 0] += r
    indices[0, :] += c
    # cv2 imread returns B, G, R (wtf?) so reverse it
    RGB = np.reshape(region[:, :, 2::-1], (-1, 3))
    # add indices and RGB next to eachother to get feature
    feature = np.hstack((indices, RGB))
    X = np.array(feature, ndmin=2, dtype=float)
    X -= mu[np.newaxis]
    return (np.divide(np.matmul(X.T, X), float(len(feature)))).squeeze()
    c = np.zeros((5, 5))
    for r in mat:
        r = np.array(r)[np.newaxis]
        c += np.multiply((r-mu), np.transpose(r-mu))
    return c / len(mat)
    # c = np.cov(mat, bias=False, rowvar=False)


modelCovMatrix = [[47.917, 0, -146.636, -141.572, -123.269],
                  [0, 408.250, 68.487, 69.828, 53.479],
                  [-146.636, 68.487, 2654.285, 2621.672, 2440.381],
                  [-141.572, 69.828, 2621.672, 2597.818, 2435.368],
                  [-123.269, 53.479, 2440.381, 2435.368, 2404.923]]
# %%
tgt = np.array(cv2.imread('target.jpg'))
limits = (tgt.shape[0]-70, tgt.shape[1]-24)
# make a list of all possible 70r, 24c regions
regions = np.array([tgt[r:r+70, c:c+24]
                   for r, c in np.ndindex(limits)])
# extract r, c from 3d array
indices = np.array(list(np.ndindex(tgt[:, :, 0].shape)))
# get the RGB values for the index
# cv2 imread returns B, G, R (wtf?) so reverse it
RGB = np.reshape(tgt[:, :, 2::-1], (-1, 3))
# combine the indices and RGB values
mu = np.hstack((indices, RGB)).mean(axis=0)


# %%
start = timer()
C = []
for r, c in np.ndindex(limits):
    C.append(cov(tgt[r:r+70, c:c+24], mu, r, c))
end = timer()
print(end-start)
pprint(C[0])
# %%


def manifold(model, cand):
    eig = linalg.eigh(model, cand, eigvals_only=True)
    return np.sqrt(np.sum(np.square(np.log2(eig, out=np.zeros_like(eig), where=(eig != 0)))))


# go through all pixels and calculate the manifold dist against the model
# ndindex iterates col first then row, so the indices will match up with the cov array
match_dist = np.zeros(limits)
for i, (r, c) in enumerate(np.ndindex(limits)):
    match_dist[r][c] = manifold(modelCovMatrix, C[i])

# %%
plt.imshow(match_dist)
print(match_dist)
# %%


def show_k_matches(img, k):
    plt.imshow(img)
    ax = plt.gca()
    ax.imshow(img)
    for i in range(k):
        i_min = np.unravel_index(np.argpartition(
            match_dist, i, axis=None)[i], match_dist.shape)
        temp = np.argpartition(
            match_dist, i, axis=None)[i]
        print(C[temp])
        print(match_dist[i_min])
        print(i_min)
        rect = patches.Rectangle(
            (i_min[1], i_min[0]), 24, 70, linewidth=1, edgecolor='r', facecolor="none", zorder=100)
        ax.add_patch(rect)
    plt.show()


show_k_matches(tgt, 1)

# It was interesting to see the intensity of the riemmanian mainfold across the image. In the places that were darker the distance is smaller. This was verified when I showed the box and it encompassed the darkest area of the diff image.
# %%
