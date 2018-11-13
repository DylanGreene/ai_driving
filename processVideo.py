"""
processVideo.py
Date: 9 November 2018
This file will contain the entire pipeline for processing a video to determine the locations of the lane lines.
It will split the video into separate images and then determine the location of the lanes in each before regenerating a video with
the lane locations added.
"""
import numpy as np
import cv2
import laneDetect
import shiftPerspective
import preprocess

if __name__ == "__main__" :

    ########## SPLIT VIDEO INTO FRAMES ##########
    # Load input video
    drivingVideo = cv2.VideoCapture('test.mp4')
    # Define codec and initialize videowriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 3 = FrameWidth; 4 = FrameHeight; 5 = FPS
    output = cv2.VideoWriter("outVid.avi", fourcc, drivingVideo.get(5), (int(drivingVideo.get(3)), int(drivingVideo.get(4))))

    while (drivingVideo.isOpened()):
        # Read the next frame
        ret, frame = drivingVideo.read()
        workingCopy = frame.copy()

        ########## PERSPECTIVE TRANSFORM ##########
        workingCopy = shiftPerspective.shift_perspective(workingCopy, 0)

        ########## PRE PROCESSING ##########
        workingCopy = preprocess.preprocess(workingCopy)

        ########## LANE DETECTION ##########
        mask = laneDetect.laneDetect(workingCopy)


        ########## LANE AREA OVERLAY ##########
        mask = shiftPerspective.shift_perspective(mask, 1)

        newFrame = cv2.addWeighted(frame, 1, mask, 0.3, 0)

        #cv2.imshow('image',newFrame)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        ########## ADD IMAGE TO OUTPUT VIDEO ##########
        output.write(newFrame)

    ########## RELEASE MEMORY ##########
    drivingVideo.release()
    output.release()
