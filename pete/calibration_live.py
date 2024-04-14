# import sys
# sys.path.append('/home/pete/.local/lib/python3.8/site-packages')
# sys.path.append('/usr/local/lib/python3.8/dist-packages')

import numpy as np
import cv2 as cv
import glob
import pickle



# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
cameraMatrix, dist = pickle.load(open("calibration.pkl","rb"))

############## UNDISTORTION #####################################################

img = cv.imread('calibrationImages/guvcview_image-A.jpg')
h,  w = 1080, 1920
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

print("first")

cv.namedWindow("preview")

vc = cv.VideoCapture(0, cv.CAP_V4L2)
vc.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
vc.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

raspi = cv.VideoCapture('tcpclientsrc host=192.168.0.248 port=5000 ! gdpdepay ! rtph264depay ! avdec_h264 ! videoconvert ! appsink sync=0 drop=1',cv.CAP_GSTREAMER)

# vc = cv.VideoCapture('v4l2src device=/dev/video0 ! video/x-raw,framerate=60/1,width=1920,height=1080 ! videoconvert !  videoflip method=rotate-180 ! video/x-raw, format=BGR ! appsink sync=0 drop=1', cv.CAP_GSTREAMER)

print(roi)

# undistort
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)

print("second")

# if vc.isOpened(): # try to get the first frame
#     rval, frame = vc.read()
# else:
#     rval = False

if raspi.isOpened(): # try to get the first frame
    rval1, frame1 = raspi.read()
else:
    rval1 = False
    

while rval1:
    # dst = frame
    # dst = cv.remap(frame, mapx, mapy, cv.INTER_LINEAR)

    # # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w -1]
    
    # final = cv.vconcat([dst,frame1])
    # cv.imshow("preview", cv.resize(final, (0,0), fx=0.5, fy=0.5))
    frame1 = frame1[151:151+670, 55:55+1186] # crop to correct size
    cv.imshow("preview",frame1)
    
    # rval, frame = vc.read()
    rval1, frame1 = raspi.read()

    key = cv.waitKey(20)
    if key == 27: # exit on ESC
        break

cv.destroyWindow("preview")

vc.release()