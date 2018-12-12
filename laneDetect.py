"""
laneDetect.py
This file contains a script that will detect lane lines in an image.
It will take an image of the road that has undergone a perspective transform and image preprocessing and determine the location of lane lines in the image.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
This function will take a transformed and processed image and return a polygon corresponding to the polynomial best fit with degree 2 for each lane line.
"""
def laneDetect(img) :
    # Image passed to this function is assumed to have been transformed and preprocessed

    ########### FIND START POINT ##########
    # Begin by taking vertical sums using numpy
    row_sum = np.sum(img,axis=0)


    # Find the two peaks
    # Split into halves by number of columns
    mid = row_sum.shape[0]/2
    histLeft = row_sum[:mid]
    histRight = row_sum[mid:]

    # Use argmax to find locations of maximums for each side (each lane)
    maxLoc = [np.argmax(histLeft), (np.argmax(histRight) + mid)]
    max = [row_sum[maxLoc[0]],row_sum[maxLoc[1]]]

    # Show histogram and peaks
    #plt.plot(row_sum)
    #plt.plot(maxLoc, max, 'r+')
    #plt.show()

    ########## DETECT PIXELS FOR EACH LINE ##########
    # Need to use a window search to make sure noise is ignored (take a large enough sample of the image)
    # Determine the window size based on the image size
    numberOfWindows = 10
    windowHeight = img.shape[0]/numberOfWindows
    windowWidth = img.shape[1]/8 # can set tweak this to change performance

    # Set the starting centerpoints and vertical positions
    cpLeft = maxLoc[0]
    cpRight = maxLoc[1]
    verticalPos = img.shape[0]

    # Load nonzero indices
    nonZeroRow,nonZeroCol = np.nonzero(img)

    # Initialize lists of points
    leftRows = []
    leftCols = []
    rightRows = []
    rightCols = []

    # Loop through windows, adding points to lists
    for i in range(numberOfWindows) :
        # Track new points to find average later (so the window col position can track the lane)
        newLeftRow = []
        newLeftCol = []
        newRightRow = []
        newRightCol = []

        for j in range(len(nonZeroRow)) :
            # Loop through non zero points, add to new trackers
            # Check if in row space of windows
            if ((nonZeroRow[j] < verticalPos) and (nonZeroRow[j] > (verticalPos - windowHeight))) :
                # Check if in left window
                if ((nonZeroCol[j] > (cpLeft - windowWidth/2)) and (nonZeroCol[j] < (cpLeft + windowWidth/2))) :
                    newLeftRow.append(nonZeroRow[j])
                    newLeftCol.append(nonZeroCol[j])
                # Check if in right window
                if ((nonZeroCol[j] > (cpRight - windowWidth/2)) and (nonZeroCol[j] < (cpRight + windowWidth/2))) :
                    newRightRow.append(nonZeroRow[j])
                    newRightCol.append(nonZeroCol[j])

        # If there are enough points in the window (i.e. more than 50), change the centerpoints
        if (len(newLeftCol) > 50) :
            cpLeft = np.mean(newLeftCol)
        if (len(newRightCol) > 50) :
            cpRight = np.mean(newRightCol)

        # Update vertical position
        verticalPos -= windowHeight

        # Add points to overall lists
        leftRows.extend(newLeftRow)
        leftCols.extend(newLeftCol)
        rightRows.extend(newRightRow)
        rightCols.extend(newRightCol)

    """
    # Test recognition
    newImg = np.zeros((img.shape[0],img.shape[1]), np.uint8)
    newImg[leftRows, leftCols] = 255
    newImg[rightRows, rightCols] = 255
    cv2.imshow('image',newImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    ########## FIND BEST FIT POLYNOMIALS ##########
    # Can use numpy polyfit
    leftFit = np.polyfit(leftRows, leftCols, 2)
    rightFit = np.polyfit(rightRows, rightCols, 2)


    # Plot best fits
    row = np.asarray(range(img.shape[0]))
    colL = leftFit[0] * (row**2) + leftFit[1] * row + leftFit[2]
    colL = colL.astype(int)

    colR = rightFit[0] * (row**2) + rightFit[1] * row + rightFit[2]
    colR = colR.astype(int)

    newChannel = np.zeros((img.shape[0],img.shape[1]), np.uint8)
    mask = np.dstack((newChannel, newChannel, newChannel))

    ########## GENERATE POLYGON ##########
    # convert points to polygon
    left = np.array([np.transpose(np.vstack([colL, row]))])
    right = np.array([np.flipud(np.transpose(np.vstack([colR, row])))])
    pts = np.hstack((left, right))

    cv2.fillPoly(mask, np.int_([pts]), (0, 0, 255))

    #cv2.imshow('image',mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    ########## RETURN MASK ##########
    return mask


if __name__ == '__main__' :
    # Set to 0 for grayscale
    img = cv2.imread("sampleCurve.png", 0)
    # Call lane detection function
    mask = laneDetect(img)
