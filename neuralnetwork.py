"""
Neural Network-based classifier, used to predict the arithmetical symbols
(digits and operators) from cropped images that contain a single symbol each
"""

import numpy as np
import cv2 as cv
import utils

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import alexnet

from PIL import Image


# 'Global' classifier variables
NET = None
NN_PATH = './NN.pth'
SYMBOLS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '/', '=', '-', '*', '+']
cont = 0                # TODO: riga da rimuovere

def init():
    """
    Initialize the classifier, by loading the neural network model
    and setting some of the required parameters
    """

    # Define a new NN model and store it as a global variable
    global NET
    NET = alexnet()

    # Change the last layer of AlexNet to output 15 classes
    NET.classifier[6] = nn.Linear(4096, len(SYMBOLS))

    # Load the already-trained model
    NET.load_state_dict(torch.load(NN_PATH, map_location=torch.device('cpu')))

    # Set the evaluation mode
    NET.eval()


def predict_symbol(image_cv):
    """
    Given a certain image containing a symbol, it performs some pre-processing
    (thresholding and morphological operations) before predicting the class label, i.e.
    the type of symbol, through the usage of a pre-trained neural network (AlexNet)
    :param image_cv: an RGB image containing a symbol
    :return: predicted label, i.e. predicted symbol
    """
    if image_cv is None:
        return None

    # Convert to grayscale for pre-processing
    img_proc = utils.rgb_to_gray(image_cv)

    # Apply thresholding to the image, separating the background from the symbol
    img_proc = cv.adaptiveThreshold(img_proc, 255,
                                    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv.THRESH_BINARY, 15, 2)

    global cont                                                     # TODO: riga da rimuovere
    cv.imwrite("./aaa_primaDelPreproc"+str(cont)+".jpg", img_proc)     # TODO: riga da rimuovere


    # Define a kernel for morphological operations
    kernel = np.array(
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]],
        np.uint8)

    # Apply closing operator to remove noise
    img_proc = cv.morphologyEx(img_proc, cv.MORPH_CLOSE, kernel)

    # Fit the image in a squared area with a 8px margin on each side (without resizing)
    height = image_cv.shape[0]
    width = image_cv.shape[1]
    size = max(height, width) + 16

    # Create the final image canvas and fill it with solid white
    image = np.zeros((size, size), dtype=np.uint8)
    image[:, :] = 255

    # Write the thresholded image on the white canvas
    y_start = int((size-height)/2)
    x_start = int((size-width)/2)
    y_end = y_start + height
    x_end = x_start + width
    image[y_start:y_end, x_start:x_end] = img_proc

    cv.imwrite("./aaa_primaDellaNN"+str(cont)+".jpg", image)        # TODO: riga da rimuovere
    cont+=1                                                         # TODO: riga da rimuovere

    # Convert image into PIL format
    image = Image.fromarray(image).convert("RGB")

    # Define transforms for the prediction phase
    eval_transform = transforms.Compose(
        [transforms.Resize(224),                                    # Resizes short size of the PIL image to 256
         transforms.ToTensor(),                                     # Turn PIL Image to torch.Tensor
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))     # Normalizes tensor with mean and standard deviation
        ])

    # Prepare the image for the network
    image = eval_transform(image).float()
    image = image.unsqueeze(0)

    # Forward the image through the network
    outputs = NET(image)

    # Get prediction (index in the SYMBOLS array)
    pred = torch.argmax(outputs)
    label = SYMBOLS[pred]
    print("Predicted symbol --> {}".format(label))

    return label
