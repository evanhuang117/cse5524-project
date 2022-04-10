# %%
from timeit import default_timer as timer
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
from math import ceil, floor,  pi, sqrt
import scipy.ndimage

# %%


def sse(a, b):
    dist = ((a[:, 0] - b[:, 0]) ** 2) + ((a[:, 1] - b[:, 1]) ** 2)
    return np.sum(dist)


# %% 1.
point3d = np.loadtxt('3Dpoints.txt')
point2d = np.loadtxt('2Dpoints.txt')

A = []
for r in range(len(point3d)):
    xw = point3d[r][0]
    yw = point3d[r][1]
    zw = point3d[r][2]
    xi = point2d[r][0]
    yi = point2d[r][1]
    A.append([xw, yw, zw, 1, 0, 0, 0, 0, -xw*xi, -yw*xi, -zw*xi, -xi])
    A.append([0, 0, 0, 0, xw, yw, zw, 1, -xw*yi, -yw*yi, -zw*yi, -yi])
A = np.array(A)
# eigh is sorted ascending
eig_val, eig_vect = np.linalg.eigh(A.T @ A)
p = eig_vect[:, 0].reshape((3, 4))

# 2.
# convert the 3d points to homog by adding a col of ones
Xw = np.hstack((point3d, np.ones((point3d.shape[0], 1)))).T
# use p to transform points and then make it inhomogenous
P = p @  Xw
P = (P / P[2, :])[:2].T
print(f'Camera calibration SSE: {sse(P, point2d)}')
# %% 3.
raw = np.loadtxt('homography.txt', delimiter=',')
im1 = raw[:, :2]
im2 = raw[:, 2:]

x1 = im1[:, 0]
y1 = im1[:, 1]
x2 = im2[:, 0]
y2 = im2[:, 1]
x1_bar = np.average(x1)
y1_bar = np.average(y1)
x2_bar = np.average(x2)
y2_bar = np.average(y2)

s1 = sqrt(2) / (1/len(im1) *
                np.sum(np.sqrt((x1-x1_bar) ** 2 + (y1 - y1_bar) ** 2)))
s2 = sqrt(2) / (1/len(im2) *
                np.sum(np.sqrt((x2-x2_bar) ** 2 + (y2 - y2_bar) ** 2)))
x1_hat = s1*(x1-x1_bar)
y1_hat = s1*(y1-y1_bar)
x2_hat = s2*(x2-x2_bar)
y2_hat = s2*(y2-y2_bar)

t1 = np.array([[s1, 0, -s1*x1_bar],
               [0, s1, -s1*y1_bar],
               [0, 0, 1]])
t2 = np.array([[s2, 0, -s2*x2_bar],
               [0, s2, -s2*y2_bar],
               [0, 0, 1]])
A = []
for r in range(len(raw)):
    x = raw[r][0]
    y = raw[r][1]
    xp = raw[r][2]
    yp = raw[r][3]
    A.append([x, y, 1, 0, 0, 0, -x*xp, -y*xp, -xp])
    A.append([0, 0, 0, x, y, 1, -x*yp, -y*yp, -yp])
A = np.array(A)
# eigh is sorted ascending
eig_val, eig_vect = np.linalg.eigh(A.T @ A)
h_squiggle = eig_vect[:, 0].reshape((3, 3))
# reverse standardization
h = np.linalg.inv(t2) @ h_squiggle @ t1


# 4. calculate projected points for img2

# convert the points to homog by adding a col of ones
im1_h = np.hstack((im1, np.ones((im1.shape[0], 1)))).T
# transform points and make inhomogenous
p2 = h@im1_h
sldfj = h@im1_h
p2 = np.linalg.inv(h) @ im1_h
p2 = (p2 / p2[2, :])[:2].T
plt.gca().scatter(p2[:, 0], p2[:, 1], label='projected')
plt.gca().scatter(im2[:, 0], im2[:, 1], label='actual')
plt.gca().legend()

# 5.
p2_sse = np.sum((p2-im2) ** 2)
plt.gca().set_title(f'Homography SSE: {p2_sse}')
plt.show()


# %%
