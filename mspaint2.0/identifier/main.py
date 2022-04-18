import cv2
from cv2 import threshold
import numpy as np



if __name__ == "__main__":

	threshold = 40
  
	# define a video capture object
	vid = cv2.VideoCapture(0)
	ret, frame = vid.read()

	# 480, 640, 3
	# many of the points below are hardcoded
	# for them to work make sure the image is of the above side before 
	# entering the loop
	height, width, d = frame.shape
	print(height, width)
	

	#this should turn drawing area into square
	line_x = width - height
	box_width = 50

	sync_finger = None
	sync_frame = None
	while(True):
		
		# Capture the video frame
		# by frame
		ret, frame = vid.read()
		frame = cv2.flip(frame, 1)
		c_frame = frame
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		# the 'q' button is set as the
		# quitting button you may use anyq
		# desired button of your choice
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		# sync with background by pressing 's'
		if cv2.waitKey(1) & 0xFF == ord('s'):
			sync_frame = frame[:, line_x:]
			print(sync_frame.shape)


		if sync_frame is not None:

			# process frame here
			cut_frame =  frame[:, line_x:]
			cut_frame = cv2.subtract(sync_frame, cut_frame)
			
			im_bw = cv2.threshold(cut_frame, threshold, 255, cv2.THRESH_BINARY)[1]

			cv2.imshow('bin', im_bw)


		# Display the resulting frame
		cv2.line(c_frame, (line_x, -10), (line_x, height + 10), (0, 255, 0), thickness=3)
		if sync_finger is None and sync_frame is not None:
				cv2.rectangle(c_frame, (375, 215), (425, 265), (0, 200, 200), 5)
		cv2.imshow('frame', c_frame)






	# After the loop release the cap object
	vid.release()
	# Destroy all the windows
	cv2.destroyAllWindows()