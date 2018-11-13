"""
preprocess.py
This function will take a bird's eye view image and threshold it such that only the lane lines are left.
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2

def preprocess(frame) :

    ########## GET NEW COLOR SPACES ##########

    # Extract b channel
    labImage = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    bImage = cv2.split(labImage)[0]
    # Extract L channel
    luvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
    lImage = cv2.split(luvImage)[0]

    # Add together before thresholding
    blImage = bImage + lImage

    ########## THRESHOLD ##########
    # Calculate threshold
    ret, thres = cv2.threshold(blImage, 127, 255, cv2.THRESH_BINARY)
    # Take inverse
    thres = cv2.bitwise_not(thres)
    
    return thres
