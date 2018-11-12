#! /usr/local/bin/python

import matplotlib.pyplot as plt
import numpy as np
import cv2



def shift_perspective(frame):

  IMG_SIZE = frame.shape[::-1][1:]
  OFFSET = 300

  PRES_SRC_PNTS = np.float32([
      (590, 411), # Top-left corner
      (390, 605), # Bottom-left corner
      (950, 605), # Bottom-right corner
      (687, 411) # Top-right corner
  ])

  PRES_DST_PNTS = np.float32([
      [OFFSET, 0], 
      [OFFSET, IMG_SIZE[1]],
      [IMG_SIZE[0]-OFFSET, IMG_SIZE[1]], 
      [IMG_SIZE[0]-OFFSET, 0]
  ])



  frame_cp = frame.copy()
  M = cv2.getPerspectiveTransform(PRES_SRC_PNTS, PRES_DST_PNTS)
  M_INV = cv2.getPerspectiveTransform(PRES_DST_PNTS, PRES_SRC_PNTS)
  warped = cv2.warpPerspective(frame, M, IMG_SIZE, flags=cv2.INTER_LINEAR)

  return warped


if __name__ == '__main__':
  frame = cv2.imread('./frames/589.png')
  



