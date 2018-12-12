"""
processVideo.py
Date: 9 November 2018
This file will contain the entire pipeline for processing a video to determine the locations of the lane lines.
It will split the video into separate images and then determine the location of the lanes in each before regenerating a video with
the lane locations added.
"""
import os
import numpy as np
import cv2
import semanticSegmentation as ss
from PIL import Image
from matplotlib import pyplot as plt

if __name__ == "__main__" :

    # Initialize dict of objects and colors for drawing object boxes
    labels = {}

    # Prep Vars for DeepLab
    model_dir = './model_dir'
    _TARBALL_NAME = 'deeplab_model.tar.gz'
    download_path = os.path.join(model_dir, _TARBALL_NAME)
    _FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    # Set up model
    model = ss.DeepLabModel(download_path)

    ########## SPLIT VIDEO INTO FRAMES ##########
    # Load input video
    drivingVideo = cv2.VideoCapture('test.mp4')
    # Define codec and initialize videowriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 3 = FrameWidth; 4 = FrameHeight; 5 = FPS
    fps = drivingVideo.get(5)
    output = cv2.VideoWriter("outVid.avi", fourcc, fps, (int(drivingVideo.get(3)), int(drivingVideo.get(4))))

    # track frame seconds
    frameNumber = 0

    while (drivingVideo.isOpened()):
        # Display total time calculated so far
        print(frameNumber / fps)

        # Read the next frame
        ret, frame = drivingVideo.read()
        workingCopy = frame.copy()
        frameNumber += 1
        
        cv2.imwrite("./frames/"+str(frameNumber)+".jpg", frame)

        ########## OBJECT DETECTION WITH DeepLab ##########
        #pil_im = Image.fromarray(frame)
        #resized_im, seg_map = model.run(pil_im)
        #newFrame = ss.vis_segmentation(resized_im, seg_map)

        #plt.imshow(newFrame)
        # Convert RGB to BGR 
        #cv2.imshow('image',newFrame)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        ########## ADD IMAGE TO OUTPUT VIDEO ##########
        #output.write(newFrame)

    ########## RELEASE MEMORY ##########
    drivingVideo.release()
    output.release()
