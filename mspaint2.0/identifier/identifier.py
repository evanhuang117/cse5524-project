
import cv2
from cv2 import imshow
import numpy as np

import time


"""
Take in a binary image where white is background and black is subject

returns x, y coordinate of finger tip and amount of finger held up
"""
def get_fingertip(image):
    

    side = incoming_side(image)






"""
Take in a binary image where white is background and black is subject

kwarg MIN_COUNT is minimum ount required to say arm comes from a side
Used in noisy environments

return approx direction finger is pointing

"""
def incoming_side(image, MIN_COUNT=15):
    

     # minimum pixels on a side to have it count
    height, width = image.shape[:2]
    sides = {"top": [], "bottom": [], "left": [], "right": []}

    points_from = [0, 0, 0, 0]

    # prepare constants to traverse boundary
    top_row = 0
    bottom_row = height - 1
    left_col = 0
    right_col = width - 1


    # Go around border and collect active pixels touching edge of image

    for i in range(width):
    
        
        if image[top_row, i] == 0:
            sides["top"].append((top_row, i))

        
        if image[bottom_row, i] == 0:
            sides["bottom"].append((bottom_row, i))


    for i in range(height):
        
        
        if image[i, left_col] == 0:
            sides["left"].append((i, left_col))

        
        if image[i, right_col] == 0:
            sides["right"].append((i, right_col))


    for k, v in sides.items():
        if len(v) > MIN_COUNT:
            if k == "top":
                

