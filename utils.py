import cv2 as cv


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


def float_to_str(value):
    """
    Format a float value into a string, stripping away trailing zeros
    :param value: the float value to be formatted
    :return: the formatted string representing the value
    """
    return ('%.15f' % value).rstrip('0').rstrip('.')
