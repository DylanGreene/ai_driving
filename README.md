# Semantic Segmentation Approach on Road Scenes for Autonomous Driving  
Notre Dame CSE 40171 | 12 December 2018 | Joe McGrath, Jack Takazawa, Paul Brunts, Dylan Greene

This project implements two different segmentation approaches for autonomous driving.  The first is a modular approach, using image processing techniques to determine the location of the lane lines followed by object detection using a neural network.  The second is a pixel level semantic segmentation approach, using neural networks in TensorFlow to determine road scene segmentation.

## Requirements  
- [OpenCV v3.4](https://opencv.org)
- [Python 2.7](https://www.python.org)
- [numpy](https://www.numpy.org)  

## Modular Segmentation Implementation
This portion of the project implements an image processing pipeline that first recognizes the lane line position using image transforms and then recognizes other objects in the image using the YOLOv3 neural network.  The function will output the current processing time in seconds so that you may track the current length of the output video.  If you wish to change the input or output videos, the filenames may be changed in processVideo.py.  Note: the program currently assumes an input video that is 1280x720. If you wish to use an input video with different dimensions, the region of interest in shiftPerspective.py will need to be changed.
Usage: python processVideo.py
## Pixel Level Semantic Segmentation
