# %%
from timeit import default_timer as timer
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
from math import ceil, floor,  pi, sqrt
import scipy.ndimage

# %%


def gaussDeriv2D(sigma):
    x = np.array(range(0, 6*ceil(sigma)))
    y = np.transpose(x)
    Gx = ((x-3*ceil(sigma))/(2*pi*sigma**4)) * \
        (np.exp(-(((x-3*ceil(sigma))**2+(y-3*ceil(sigma))**2)/(2*sigma**2))))
    Gy = ((y-3*ceil(sigma))/(2*pi*sigma**4)) * \
        (np.exp(-(((x-3*ceil(sigma))**2+(y-3*ceil(sigma))**2)/(2*sigma**2))))
    return [Gx, Gy[np.newaxis]]


def blur(im, a):
    Gx, Gy = gaussDeriv2D(a)
    Gx = Gx[:, None]
    gxIm = scipy.ndimage.correlate(im, Gx, mode='nearest')
    gyIm = scipy.ndimage.correlate(im, Gy, mode='nearest')
    blurred = np.sqrt(gxIm**2 + gyIm**2)
    return blurred


def cov(region, mu, r, c):
    indices = np.array(list(np.ndindex(region[:, :, 0].shape)))
    indices[:, 0] += r
    indices[0, :] += c
    RGB = np.reshape(region[:, :, 2], (-1, 3))
    # add indices and RGB next to eachother to get feature
    feature = np.hstack((indices, RGB))
    X = np.array(feature, ndmin=2, dtype=float)
    X -= mu[np.newaxis]
    return (np.divide(np.matmul(X.T, X), float(len(feature)))).squeeze()

# %%


checker = plt.imread('checker.png')*255
alpha = 0.05
dx, dy = gaussDeriv2D(0.7)

ix = cv2.filter2D(checker, -1, dx)
iy = cv2.filter2D(checker, -1, dy)

ix2 = np.square(ix)
iy2 = np.square(iy)
ixiy = ix @ iy

gx = cv2.GaussianBlur(ix2, (3, 3), 1)
gy = cv2.GaussianBlur(iy2, (3, 3), 1)
gxgy = cv2.GaussianBlur(ixiy, (3, 3), 1)

R = gx*gy - np.square(gxgy) - alpha*np.square(gx+gy)
# for some reason I get negative significant values instead of positive?
R = -R
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(R)
plt.gca().set_title('raw R', c='r')
threshold = np.where(R < 1e6, 0, R)
threshold = R > 1e6
plt.subplot(1, 2, 2)
plt.imshow(threshold)
plt.gca().set_title('thresholded', c='r')
print('R(16:22, 16:22)')
print(R[16:22, 16:22])

w_size = 3
limits = threshold.shape[0]-w_size, threshold.shape[1]-w_size
suppress = []
for r, c in np.ndindex(limits):
    region = threshold[r-1:r+1, c-1:c+1]
    if np.unique(region).size > 0:
        if np.max(np.unique(region)) > 0:
            suppress.append((c, r))

plt.suptitle('Harris')
plt.show()

plt.figure(figsize=(5, 5))
plt.gca().imshow(checker, cmap='gray')
plt.gca().scatter(*zip(*suppress), s=0.5, c='r')
plt.title('overlay')
plt.show()

# %% 2.


def fast(img, T):
    offsets = [(3, 0),
               (3, 1),
               (2, 2),
               (1, 3),
               (0, 3),
               (-1, 3),
               (-2, 2),
               (-3, 1),
               (-3, 0),
               (-3, -1),
               (-2, -2),
               (-1, -3),
               (0, -3),
               (1, -3),
               (-2, 2),
               (3, -1)]
    features = []
    for r, c in np.ndindex(img.shape):

        ind = [(r+dr, c+dc) for dr, dc in offsets]
        vals = np.array([img[i] for i in ind
                        if i[0] >= 0 and i[0] < img.shape[0] and i[1] >= 0 and i[1] < img.shape[1]])
        vals = np.hstack((vals, vals))

        head = 0
        tail = 0
        # while tail <= len(vals) - 9:
        #     if tail - head >= 9:
        #         features.append((c, r))
        #         break
        #     elif vals[tail] > img[r, c] + T:
        #         tail += 1
        #     else:
        #         head = tail
        #         tail = head+1

        # while tail <= len(vals) - 9:
        #     if tail - head >= 9:
        #         features.append((c, r))
        #         break
        #     elif vals[tail] < img[r, c] - T:
        #         tail += 1
        #     else:
        #         head = tail
        #         tail = head+1
        run = 0
        i = 0
        while i < len(vals):
            if run >= 9:
                features.append((c, r))
                break
            elif vals[i] > img[r, c] + T:
                i += 1
                run += 1
            else:
                i += 1
                run = 0

    return features


# %%
img = plt.imread('tower.png') * 255
plt.figure(figsize=(10, 10))
start = timer()
for i, T in enumerate([10, 20, 30, 50]):
    plt.subplot(2, 2, i+1)
    features = fast(img, T)
    plt.gca().imshow(img)
    plt.gca().scatter(*zip(*features), s=0.5, c='r')
    plt.gca().set_title(f'T={T}', c='r')
    end = timer()
    print(end-start)
    break
plt.suptitle('FAST')
# %%
