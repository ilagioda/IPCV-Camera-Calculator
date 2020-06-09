import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys
import utils


def prepare_image(image):
    """
    Image pre-processing, before running them through the neural network
    :param image: an RGB image of a symbol that has to be processed by the neural network
    :return:
    """
    if image is None:
        return

    plt.subplot(1, 3, 1).axis('on')
    plt.imshow(image)
    plt.title("Input image")

    # Convert to grayscale
    img_gray = utils.rgb_to_gray(image)

    # Scale the input image to fit in a 64x64 area (keep aspect ratio)
    # with a 4px margin on each side (actual size = 56x56)
    height = image.shape[0]
    width = image.shape[1]
    scale_factor = 56 / (height if height > width else width)
    height = np.uint(scale_factor * height)
    width = np.uint(scale_factor * width)

    # Bicubic interpolation for better results (slower)
    img_gray = cv.resize(img_gray, (width, height), interpolation=cv.INTER_CUBIC)

    # Apply thresholding to the image, separating the background from the symbol
    thresh = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 2)

    plt.subplot(1, 3, 2).axis('on')
    plt.imshow(thresh, 'gray')
    plt.title("Thresholded image")

    # Define a kernel for morphological operations
    kernel = np.array(
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]],
        np.uint8)

    # Apply closing operator to remove noise
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    # Create a 64x64 image and fill it with solid white
    image_prepared = np.zeros((64, 64), dtype=np.uint8)
    image_prepared[:, :] = 255

    # Write the thresholded image on the white-filled 64x64 canvas
    image_prepared[
        32-np.uint(height/2):32+np.uint(height/2),
        32-np.uint(width/2):32+np.uint(width/2)] \
        = thresh

    plt.subplot(1, 3, 3).axis('on')
    plt.imshow(image_prepared, 'gray')
    plt.title("Post-processed image")
