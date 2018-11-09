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

if __name__ == "__main__" :

    ########## SPLIT VIDEO INTO FRAMES ##########
    # Load input video
    drivingVideo = cv2.VideoCapture('sampleVideo.mp4')
    # Define codec and initialize videowriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 3 = FrameWidth; 4 = FrameHeight; 5 = FPS
    output = cv2.VideoWriter("outVid.avi", fourcc, drivingVideo.get(5), (int(drivingVideo.get(3)), int(drivingVideo.get(4))))

    while (drivingVideo.isOpened()):
        # Read the next frame
        ret, frame = drivingVideo.read()

        ########## PERSPECTIVE TRANSFORM ##########

        ########## PRE PROCESSING ##########

        ########## LANE DETECTION ##########

        ########## LANE AREA OVERLAY ##########

        ########## ADD IMAGE TO OUTPUT VIDEO ##########
        output.write(newFrame)

    ########## RELEASE MEMORY ##########
    drivingVideo.release()
    output.release()
