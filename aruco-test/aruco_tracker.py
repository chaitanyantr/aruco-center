
import numpy as np
import cv2
import cv2.aruco as aruco
import glob

cap = cv2.VideoCapture(0)

####---------------------- CALIBRATION ---------------------------
# termination criteria for the iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# checkerboard of size (7 x 6) is used
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# iterating through all calibration images
# in the folder
images = glob.glob('calib_images/*.jpg')

for fname in images:
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# find the chess board (calibration pattern) corners
	ret, corners = cv2.findChessboardCorners(gray,(9,6),None)

	# if calibration pattern is found, add object points,
	# image points (after refining them)
	# print(ret)
	if ret == True:
		objpoints.append(objp)

		# Refine the corners of the detected corners
		corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
		imgpoints.append(corners2)

		# Draw and display the corners
		img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
		
		
	   
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

###------------------ ARUCO TRACKER ---------------------------
while (True):
	ret, frame = cap.read()

	# operations on the frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# set dictionary size depending on the aruco marker selected
	aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

	# detector parameters can be set here (List of detection parameters[3])
	parameters = aruco.DetectorParameters_create()
	parameters.adaptiveThreshConstant = 10

	# lists of ids and the corners belonging to each id
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
	
	c1x = 0.0
	c1y = 0.0
	c2x = 0.0
	c2y = 0.0

	fid=[]
	if(len(corners) == 1 ):

		if(ids[0] == 1):
			if(len(corners[0]) != 0):
				fid.insert(0,corners[0])
				#print(fid[0])
				#print('id-01',corners[0])
				

		if(ids[0] == 34):
			if(len(corners[0]) != 0):
				fid.insert(1,corners[0])
				#print(fid[0])
				#print('id-34',corners[0])			
				#print('gg',fid[0])
				#print(fid[0][1])

	elif(len(corners) == 2 ):

		if(ids[0] == 1):
			if(len(corners[0]) != 0):
				fid.insert(0,corners[0])
				#print(fid[0])
				#print('id-01',corners[0])
				#print('1-id')
				#print("f1",fid[0])

				# print('0',fid[0][0][0][1])
				# print('1',fid[0][0][1][1])
				# print('3',fid[0][0][3][1])

				c1x = (fid[0][0][0][0] + fid[0][0][1][0] + fid[0][0][2][0]  + fid[0][0][3][0])/4
				c1y = (fid[0][0][0][1] + fid[0][0][1][1] + fid[0][0][2][1]  + fid[0][0][3][1])/4
				
				#print('cx',c1x,'cy',c1y)


		if(ids[1] == 34):
			if(len(corners[1]) != 0):
				fid.insert(1,corners[1])
				#print(fid[1])
				#print('id-34.',corners[1])

				#print('34-id')
				#print('f2',fid[1])

				# print('0',fid[1][0][0][1])
				# print('1',fid[1][0][1][1])
				# print('3',fid[1][0][3][1])

				c2x = (fid[1][0][0][0] + fid[1][0][1][0] + fid[1][0][2][0]  + fid[1][0][3][0])/4
				c2y = (fid[1][0][0][1] + fid[1][0][1][1] + fid[1][0][2][1]  + fid[1][0][3][1])/4
				
				#print('cx',c2x,'cy',c2y)

		# if(ids[0] == 34):
		# 	if(len(corners[0]) != 0):
		# 		fid.insert(1,corners[0])
		# 		#print(fid[0])
		# 		#print('id-34',corners[0])

		# if(ids[1] == 1):
		# 	if(len(corners[1]) != 0):
		# 		fid.insert(0,corners[1])
		# 		#print(fid[1])
		# 		#print('id-01.',corners[1])


	font = cv2.FONT_HERSHEY_SIMPLEX

	# check if the ids list is not empty
	# if no check is added the code will crash
	if np.all(ids != None):

		# estimate pose of each marker and return the values
		# rvet and tvec-different from camera coefficients
		rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
		#(rvec-tvec).any() # get rid of that nasty numpy value array error

		#for i in range(0, ids.size):
			# draw axis for the aruco markers
			#aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)
			#print(rvec,tvec)

		# draw a square around the markers
		aruco.drawDetectedMarkers(frame, corners)


		# code to show ids of the marker found
		strg = ''
		for i in range(0, ids.size):
			strg += str(ids[i][0])+', '

		cv2.putText(frame, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

		#print('ids',ids)

	else:
		# code to show 'No Ids' when no markers are found
		cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

	cv2.circle(frame,(int(c1x),int(c1y)), 5, (0,255,0), -1)
	cv2.circle(frame,(int(c2x),int(c2y)), 5, (0,255,0), -1)
	# display the resulting frame
	cv2.imshow('frame',frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# References
# 1. https://docs.opencv.org/3.4.0/d5/dae/tutorial_aruco_detection.html
# 2. https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
# 3. https://docs.opencv.org/3.1.0/d5/dae/tutorial_aruco_detection.html
