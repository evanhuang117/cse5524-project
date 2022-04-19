
from pickle import FALSE
import cv2
from cv2 import imshow
import numpy as np

import time


"""
Take in a binary image where white is background and black is subject

returns x, y coordinate of finger tip and amount of finger held up
"""
def get_fingertip(image, test=False):
    

    sides = incoming_side(image)
    finger_tip_coords = find_point(image, sides)

    if test:
        cv2.circle(image, (finger_tip_coords[0], finger_tip_coords[1]), radius=20, color=(0, 0 ,0), thickness=2) 
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #print(finger_tip_coords)
    return finger_tip_coords





def secondary_check(image, sides):

    # constant for finger width
    finger_offset = 40
    height, width = image.shape[:2]
    top, bottom, left, right = sides[0], sides[1], sides[2], sides[3]
    finger_height = -25
    finger_width = -25


    if sum(sides) >= 1:

        if bottom:
            
            for i in range(height):
                row = image[i]
                if 255 in row:
                    finger_height = i
                    finger_width = np.median(np.where(row == 255))
                    return (int(finger_width), int(finger_height))


def find_point(image, sides):

    # constant for finger width
    finger_offset = 120
    check_constant = 20
    height, width = image.shape[:2]
    top, bottom, left, right = sides[0], sides[1], sides[2], sides[3]
    finger_height = -25
    finger_width = -25

    fingers = 0

    coords = []

    if sum(sides) >= 1:
        # test cardinal directions

        if bottom:
            # search from top
            for i in range(height):
                row = image[i]
                if 255 in row:
                    finger_height = i
                    finger_width = int(np.median(np.where(row == 255)))

                    left_check = finger_width - finger_offset

                    coords.append((int(finger_width), int(finger_height)))

                    if left_check > finger_offset and height > 2 * finger_offset:
                        check_im = image[finger_height:finger_height + finger_offset, left_check - finger_offset:left_check]
                        other_coords = secondary_check(check_im, sides)
                        if other_coords is not None:
                            new_coords = (other_coords[0] + (left_check - finger_offset), other_coords[1] + finger_height)
                            print(f"left fingertip: {new_coords}")

                            coords.append(new_coords)

                    right_check = finger_width + finger_offset
                    if right_check < width - check_constant and height > 2 * finger_offset:
                        check_im = image[finger_height:finger_height + finger_offset, right_check: right_check + finger_offset]
                        other_coords = secondary_check(check_im, sides)

                        if other_coords is not None:
                            new_coords = (other_coords[0] + right_check, other_coords[1] + finger_height)
                            #print(f"right fingertip: {new_coords}")
                            coords.append(new_coords)
                        


                    #print(f"All: {coords}")
                    break
            
        elif left:
            # search from right
            for i in range(width - 1, 0, -1):
                col = image[:,i]
                if 255 in col:
                    finger_height = np.median(np.where(col == 255))
                    finger_width = i
                    coords.append((int(finger_width), int(finger_height)))
                    break
        elif right:
            # search from left
                for i in range(width):
                    col = image[:,i]
                    if 255 in col:
                        finger_height = np.median(np.where(col == 255))
                        finger_width = i
                        coords.append((int(finger_width), int(finger_height)))
                        break
        elif top:
            # search from bottom
                for i in range(height - 1, 0, -1 ):
                    row = image[i]
                    if 255 in row:
                        finger_height = i
                        finger_width = np.median(np.where(row == 255))
                        coords.append((int(finger_width), int(finger_height)))
                        break


        

    elif sum(sides) == 2:
        # harder case check diaganol

        if bottom and right:
            # search from top left
            pass

        elif bottom and left:
            # search from top right
            pass
        elif top and right:
            # search from bottom left
            pass
        elif top and left:
            # search from bottom right
            pass



    final = [x for x in coords if x is not None]
    
    return final
    #return (int(finger_width), int(finger_height))




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

    

    # prepare constants to traverse boundary
    top_row = 0
    bottom_row = height - 1
    left_col = 0
    right_col = width - 1


    # Go around border and collect active pixels touching edge of image

    for i in range(width):
    
        if image[top_row, i] == 255:
            sides["top"].append((top_row, i))

        if image[bottom_row, i] == 255:
            sides["bottom"].append((bottom_row, i))


    for i in range(height):
        
        if image[i, left_col] == 255:
            sides["left"].append((i, left_col))

        if image[i, right_col] == 255:
            sides["right"].append((i, right_col))


    points_from = [len(sides["top"]) > MIN_COUNT, len(sides["bottom"]) > MIN_COUNT, len(sides["left"]) > MIN_COUNT, len(sides["right"]) > MIN_COUNT]
    #print(points_from)

    return points_from





if __name__ == '__main__':
    print("Running tests")

    hand = cv2.imread('testImages/point_up.jpg', 2)
    #hand = cv2.imread('testImages/point_right.jpg', 2)
    #hand = cv2.imread('testImages/point_left.jpg', 2)
    #hand = cv2.imread('testImages/point_down.jpg', 2)
    #hand = cv2.imread('testImages/point_down.jpg', 2)
    #hand = cv2.imread('testImages/down_left_easy.jpg', 2)

    video_fnam = "../../"

    '''
    cap = cv2.VideoCapture('linusi.mp4')

    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    '''

    

    #210 255 seem like good bounds here
    ret, bw = cv2.threshold(hand,  210, 255, cv2.THRESH_BINARY)

    start = time.time()
    get_fingertip(bw, test=True)
    end = time.time()

    # will not work as intended if viewing image
    #print(f"Time taken: {end - start}")


