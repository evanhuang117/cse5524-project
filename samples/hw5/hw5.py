# %%
from PIL import Image, ImageDraw
from re import M
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
from math import ceil, floor,  pi, sqrt
import scipy.ndimage
from pprint import pprint
import os

# %%
def showSample(imgs, title):
    plt.figure()
    for i, imNum in enumerate(np.linspace(0, len(imgs)-1, 4, dtype=int)):
        plt.subplot(2, 2, i+1)
        plt.imshow(imgs[imNum], cmap='gray')
    plt.suptitle(title)

# %%
# 1.
imgs = []
for i in range(1, 23):
    name = f'./data/aerobic-{i:03}.bmp'
    imgs.append(np.array(rgb2gray(cv2.imread(name))))

diffs = []
for i in range(0, len(imgs) - 1):
    diffs.append(np.abs(imgs[i+1] - imgs[i]))

diffs = [d > 0.04 for d in diffs]
showSample(diffs, 'simple, after threshold')

T = 9
medBlur = []
for img in diffs:
    medBlur.append(cv2.medianBlur(img.astype(np.uint8), T))
showSample(medBlur, 'simple, median blur')

open_close = []
for img in diffs:
    temp = scipy.ndimage.binary_opening(img) 
    temp = scipy.ndimage.binary_closing(temp) 
    temp = cv2.medianBlur(temp.astype(np.uint8), T)
    open_close.append(temp)
showSample(open_close, 'simple, area opening->closing')

open_close_blur = []
for img in diffs:
    temp = scipy.ndimage.binary_opening(img) 
    temp = scipy.ndimage.binary_closing(temp) 
    temp = cv2.medianBlur(temp.astype(np.uint8), T)
    open_close_blur.append(temp)
showSample(open_close_blur, 'simple, area opening->closing->blur')
# %%
# 2.
def similitudeMoments(img):
    m_0_0 = calc_moment(img, 0, 0)
    x_bar = calc_moment(img, 1, 0) / m_0_0
    y_bar = calc_moment(img, 0, 1) / m_0_0
    nu = []
    for i in range(4):
        for j in range(4):
            if i+j >= 2 and i+j <= 3:
                nu.append(
                    (calc_moment(img, i, j, x_bar, y_bar) /
                     m_0_0**((i+j)/2+1), f'i={i}, j={j}')
                )
    return nu

def calc_moment(img, i, j, x_bar=0, y_bar=0):
    m = 0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            m += (x-x_bar)**i * (y-y_bar)**j * img[x][y]
    return m


# MEI
mei = np.copy(diffs)
for i in range(len(mei)-1):
    img = mei[i]
    nextImg = mei[i+1]
    mei[i+1] = np.logical_or(img, nextImg)

showSample(mei, 'MEI')

# MHI
mhi = []
for i, img in enumerate(diffs):
    tstamp = i + 2
    mhi.append(np.vectorize(lambda x: max(
        0.0, (tstamp-1.0)/21.0) if x else 0.0)(img))

plt.figure()
plt.imshow(mhi[0], cmap='gray')
for i in range(len(mhi)-1):
    img = mhi[i]
    nextImg = mhi[i+1]
    for x in range(len(img)):
        for y in range(len(img[x])):
            mhi[i+1][x][y] = max(img[x][y], nextImg[x][y])

showSample(mhi, 'MHI')

# calculate similitude moments




# %%
# 3.
box0 = Image.new('L', (101, 101), color=0)
ImageDraw.Draw(box0).rectangle([40, 6, 61, 27], fill=1)
box1 = Image.new('L', (101, 101), color=0)
ImageDraw.Draw(box1).rectangle([41, 7, 62, 28], fill=1)
box0 = np.array(box0)
box1 = np.array(box1)
plt.figure(figsize=(10,10), dpi=500)
plt.imshow(box0, cmap='gray')
plt.imshow(box1, cmap='gray')
Gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]) / 8
Gy = np.array([[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]]) / 8

fx = scipy.ndimage.correlate(box1, Gx, mode='nearest')
fy = scipy.ndimage.correlate(box1, Gy, mode='nearest')
fx= cv2.filter2D(box1.astype(np.float32), -1, Gx)
fy= cv2.filter2D(box1.astype(np.float32), -1, Gy)
plt.figure()
plt.imshow(fx)
plt.figure()
plt.imshow(fy)

ft = box1 - box0
denom = np.sqrt(fx**2+fy**2)
mag = np.nan_to_num(-ft / denom)
u = np.nan_to_num((fx / denom) * mag)
v = np.nan_to_num((fy / denom) * mag)

for x, y in np.ndindex(box1.shape):
    u1 = u[x][y]
    v1 = v[x][y]
    if v1 or u1:
        plt.quiver(y, x, v1, u1, color='red', width=0.002)


# %%
