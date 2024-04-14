import numpy as np
import cv2 as cv
import glob
import pickle

# print(cv.__version__)
# print(cv.getBuildInformation())


cv.namedWindow("preview")
# vc = cv.VideoCapture('v4l2src device=/dev/video0 ! video/x-raw,framerate=60/1,width=1920,height=1080 ! videoconvert !  videoflip method=rotate-180 ! video/x-raw, format=BGR ! appsink sync=0 drop=1', cv.CAP_GSTREAMER)
vc = cv.VideoCapture(0, cv.CAP_V4L2)

# tcpclientsrc host=192.168.0.248 port=5000 ! gdpdepay ! rtph264depay ! avdec_h264 ! videoconvert ! appsink sync=0 drop=1
vc.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
vc.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
vc.set(cv.CAP_PROP_FPS, 20)


if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    # frame = cv.resize(frame, (1280,720), cv.INTER_AREA)
    cv.imshow("preview", frame)
    rval, frame = vc.read()

    key = cv.waitKey(20)
    if key == 27: # exit on ESC
        break

cv.destroyWindow("preview")
vc.release()