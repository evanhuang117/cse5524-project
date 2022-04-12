import numpy as np
import cv2
import time

fname = "../../data/up_move_right.mp4"
fname2 = "../../data/angle_left_move_right.mp4"

THRESHOLD = 25

cap = cv2.VideoCapture(fname)

ret, prevFrame = cap.read()
prevFrame = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)

while(cap.isOpened()):
    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mei = np.abs(frame-prevFrame)

    print(mei)
    mei[mei < THRESHOLD] = 255
    #mei[mei >THRESHOLD] = 0
    

    cv2.imshow('frame',mei)

    #time.sleep(0.2)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prevFrame = frame

cap.release()
cv2.destroyAllWindows()