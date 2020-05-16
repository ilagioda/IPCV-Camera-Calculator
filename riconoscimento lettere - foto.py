from __future__ import print_function
import cv2 as cv
import numpy as np
import random as rng
import sys

rng.seed(12345)

def thresh_callback(val):
    threshold = val

    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)

    # Find contours
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Create the content of window
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(contours)):
        hull = cv.convexHull(contours[i])
        hull_list.append(hull)

        # Find the coordinates of the rectangle that is around the letter
        min_x = sys.maxsize
        min_y = sys.maxsize
        max_x = -1
        max_y = -1
        for (num, vet) in enumerate(hull):
            # print("num = {}, coord = {}".format(num, vet))
            coord = vet[0]
            x = coord[0]
            y = coord[1]
            # print("x = {}, y = {}".format(x, y))
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
        print("min_x = {}, min_y = {}, max_x = {}, max_y = {}".format(min_x, min_y, max_x, max_y))

        # Start coordinate, represents the top left corner of rectangle
        start_point = (min_x, min_y)
        # Ending coordinate, represents the bottom right corner of rectangle
        end_point = (max_x, max_y)
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        # Draw the rectangle around each letter
        drawing = cv.rectangle(drawing, start_point, end_point, color, thickness)

    # Draw contours + hull results
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours, i, color)
        cv.drawContours(drawing, hull_list, i, color)

    # Show in a window
    cv.imshow('Contours', drawing)

# Load source image
src = cv.imread("ciao.jpg")
src = cv.resize(src, (960, 540))

# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3,3))

# Create Window
source_window = 'Source'
cv.namedWindow(source_window, cv.WINDOW_NORMAL)
cv.imshow(source_window, src)
max_thresh = 255 # max threshold
thresh = 100 # initial threshold
cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
cv.waitKey()