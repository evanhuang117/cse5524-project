import cv2
from cv2 import threshold
import numpy as np


class Tracker():

	def __init__(self):
		pass








if __name__ == "__main__":

	t = Tracker()
	threshold = 40
  
	# define a video capture object
	vid = cv2.VideoCapture(0)
	ret, frame = vid.read()
	#480, 640, 3
	
	height, width, d = frame.shape
	print(height, width)
	

	#this should turn drawing area into square
	line_x = width - height


	sync_frame = None
	while(True):
		
		# Capture the video frame
		# by frame
		ret, frame = vid.read()
		frame = cv2.flip(frame, 1)
		c_frame = frame
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		# the 'q' button is set as the
		# quitting button you may use any
		# desired button of your choice
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		# sync with background
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
		cv2.imshow('frame', c_frame)






	# After the loop release the cap object
	vid.release()
	# Destroy all the windows
	cv2.destroyAllWindows()