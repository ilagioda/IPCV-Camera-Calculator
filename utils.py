"""A collection of utility functions"""

import cv2 as cv
import numpy as np


def bgr_to_gray(image):
    """
    Convert a BGR image into grayscale
    :param image: the BGR image
    :return: the same image but in grayscale
    """
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def bgr_to_rgb(image):
    """
    Convert a BGR image into RGB
    :param image: the BGR image
    :return: the same image but in the RGB color space
    """
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def bgr_to_hsv(image):
    """
    Convert a BGR image into HSV
    :param image: the BGR image
    :return: the same image but in HSV
    """
    return cv.cvtColor(image, cv.COLOR_BGR2HSV)


def hsv_to_bgr(image):
    """
    Convert a HSV image into BGR
    :param image: the HSV image
    :return: the same image but in the BGR color space
    """
    return cv.cvtColor(image, cv.COLOR_HSV2RGB)


def hsv_to_rgb(image):
    """
    Convert a HSV image into RGB
    :param image: the HSV image
    :return: the same image but in the RGB color space
    """
    return cv.cvtColor(image, cv.COLOR_HSV2RGB)


def hsv_to_gray(image):
    """
    Convert a HSV image into grayscale by keeping only the V component
    :param image: the HSV image
    :return: the same image but in grayscale
    """
    return image[:, :, 2]


def rgb_to_gray(image):
    """
    Convert a RGB image into grayscale
    :param image: the RGB image
    :return: the same image but in grayscale
    """
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)


def rgb_to_bgr(image):
    """
    Convert a RGB image into BGR
    :param image: the RGB image
    :return: the same image but in the BGR color space
    """
    return cv.cvtColor(image, cv.COLOR_RGB2BGR)


def rgb_to_hsv(image):
    """
    Convert a RGB image into HSV
    :param image: the RGB image
    :return: the same image but in HSV
    """
    return cv.cvtColor(image, cv.COLOR_RGB2HSV)


def crop(image, start, end):
    """
    Extract a portion of the provided image, by the specified coordinates
    :param image: the source image
    :param start: the top-left point (x, y) of the crop area
    :param end: the bottom-right point (x, y) of the crop area
    :return: the cropped portion of the image (or None)
    """
    if image is None:
        return None

    if start is None or len(start) != 2 or end is None or len(end) != 2:
        return None

    return image[start[1]:end[1], start[0]:end[0]]


def float_to_str(value):
    """
    Format a float value into a string, stripping away trailing zeros
    :param value: the float value to be formatted
    :return: the formatted string representing the value
    """
    return ('%.15f' % value).rstrip('0').rstrip('.')


def rectangle_center(rectangle):
    """
    Calculate the center point of a rectangle
    :param rectangle: the rectangle in the form [min_x, min_y, max_x, max_y]
    :return: (x, y) coordinates of the center (or None)
    """
    if not rectangle or len(rectangle) != 4:
        return None

    return (np.int((rectangle[2] + rectangle[0]) / 2),
            np.int((rectangle[3] + rectangle[1]) / 2))


def point_distance(point_a, point_b):
    """
    Compute the distance between 2 points in bi-dimensional space (x, y)
    :param pointA: (x, y) coordinates of the first point
    :param pointB: (x, y) coordinates of the second point
    :return: the computed distance (or None)
    """
    if not point_a or len(point_a) != 2 or not point_b or len(point_b) != 2:
        return None

    return np.sqrt(((point_a[0] - point_b[0]) ** 2) + ((point_a[1] - point_b[1]) ** 2))
