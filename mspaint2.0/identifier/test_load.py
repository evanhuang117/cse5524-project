import cv2
from cv2 import imshow
import numpy as np

import time


#hand = cv2.imread('testImages/point_up.jpg', 2)
#hand = cv2.imread('testImages/point_right.jpg', 2)
#hand = cv2.imread('testImages/point_left.jpg', 2)
#hand = cv2.imread('testImages/point_down.jpg', 2)
#hand = cv2.imread('testImages/point_down.jpg', 2)
hand = cv2.imread('testImages/down_left_easy.jpg', 2)




print(hand)
#cv2.imshow("test", hand)


height, width = hand.shape[:2]
# 210 good thresshold
ret, bw = cv2.threshold(hand,  210, 255, cv2.THRESH_BINARY)

bw_img = cv2.threshold(hand, 210, 255, cv2.THRESH_BINARY)

#cv2.imshow("b2", bw)

start = time.time()



left = False
top = False
right = False
bottom = False

sides = {"top": [], "bottom": [], "left": [], "right": []}

#top border
topP = 0
bottomP = height - 1

leftP = 0
rightP = width - 1


# Go around border and collect active pixels touching edge of image

for i in range(width):
   
    
    if bw[topP, i] == 0:
        top = True
        sides["top"].append((top, i))

    
    if bw[bottomP, i] == 0:
        bottom = True
        sides["bottom"].append((bottom, i))


for i in range(height):
    
    
    if bw[i, leftP] == 0:
        left = True
        sides["left"].append((i, leftP))

    
    if bw[i, rightP] == 0:
        right = True
        sides["right"].append((i, rightP))

print(sides['left'])

# this will be messy but it will be fast

#only intersects with the bottom. So finger tip is highest point
finger_height = -1
finger_width = []
if bottom and not left and not right and not top:

    for i in range(height):
        row = bw[i]
        if 0 in row:
            finger_height = i
            finger_width = np.median(np.where(row == 0))
            break

    
elif left and not right and not top and not bottom:
    
    for i in range(width - 1, 0, -1):
        col = bw[:,i]
        if 0 in col:
            finger_height = np.median(np.where(col == 0))
            finger_width = i
            break


elif right and not left and not top and not bottom:
    for i in range(width):
        col = bw[:,i]
        if 0 in col:
            finger_height = np.median(np.where(col == 0))
            finger_width = i
            break


elif top and not bottom and not left and not right:

    for i in range(height - 1, 0, -1 ):
        row = bw[i]
        if 0 in row:
            finger_height = i
            finger_width = np.median(np.where(row == 0))
            break



print(finger_width)
print(finger_height)

cv2.circle(bw, (int(finger_width), int(finger_height)), radius=20, color=(0, 0 ,0), thickness=2) 




end = time.time()


print(f"Time taken: {end-start}")
    

        
    



#cv2.circle(bw, left[::-1], radius=10, color=(255, 0 ,0), thickness=6) 

#cv2.circle(bw, right[::-1], radius=10, color=(255, 0 ,0), thickness=6) 





#bottom border


#left side

#right side


cv2.imshow("Image", bw)


#cv2.imwrite("test_point_left.jpg", bw)



cv2.waitKey(0)
cv2.destroyAllWindows()