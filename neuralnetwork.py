"""
Neural Network-based classifier, used to predict the arithmetical symbols
(digits and operators) from cropped images that contain a single symbol each
"""

import cv2 as cv
import numpy as np
import utils

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import alexnet

from PIL import Image


# Set of 'global' classifier variables
NUM_CLASSES = 15
NN_PATH = './NN.pth'
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
          'div', 'equal', 'minus', 'mul', 'plus']


def prepare_image(image):
    """
    Image pre-processing, before running them through the neural network
    :param image: an RGB image of a symbol that has to be processed by the neural network
    :return: the prepared and pre-processed image
    """
    if image is None:
        return None

    # Convert to grayscale
    img_proc = utils.rgb_to_gray(image)

    # Apply thresholding to the image, separating the background from the symbol
    img_proc = cv.adaptiveThreshold(img_proc, 255,
                                    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv.THRESH_BINARY, 15, 2)

    # Define a kernel for morphological operations
    kernel = np.array(
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]],
        np.uint8)

    # Apply closing operator to remove noise
    img_proc = cv.morphologyEx(img_proc, cv.MORPH_CLOSE, kernel)

    # Fit the image in a squared area with a 8px margin on each side (without resizing)
    height = image.shape[0]
    width = image.shape[1]
    size = max(height, width) + 16

    # Create the final image canvas and fill it with solid white
    image_prepared = np.zeros((size, size), dtype=np.uint8)
    image_prepared[:, :] = 255

    # Write the thresholded image on the white canvas
    y_start = int((size-height)/2)
    x_start = int((size-width)/2)
    y_end = y_start + height
    x_end = x_start + width
    image_prepared[y_start:y_end, x_start:x_end] = img_proc

    return image_prepared


def predict_symbol(image_cv):
    """
    Given a certain image containing a symbol, allows to predict the class label, i.e.
    the type of symbol, through the usage of a pre-trained neural network (AlexNet)
    :param img: an image containing a symbol
    :return: predicted label, i.e. predicted symbol
    """

    # Convert image into PIL format
    image_cv = cv.cvtColor(image_cv, cv.COLOR_BGR2RGB)
    image = Image.fromarray(image_cv).convert("RGB")

    # Define transforms for the prediction phase
    eval_transform = transforms.Compose(
        [transforms.Resize(224),                                    # Resizes short size of the PIL image to 256
         transforms.ToTensor(),                                     # Turn PIL Image to torch.Tensor
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))     # Normalizes tensor with mean and standard deviation
        ])

    # Define a new NN model
    net = alexnet()

    # Change the last layer of AlexNet to output 15 classes
    net.classifier[6] = nn.Linear(4096, NUM_CLASSES)

    # Load the already-trained model
    net.load_state_dict(torch.load(NN_PATH, map_location=torch.device('cpu')))

    # Set the evaluation mode
    net.eval()

    # Prepare the image for the network
    image = eval_transform(image).float()
    image = image.unsqueeze(0)

    # Forward the image through the network
    outputs = net(image)

    # Get prediction (index in the LABELS array)
    pred = torch.argmax(outputs)
    print("Predicted index --> {}".format(pred))

    # Retrieve the predicted label from the LABELS array
    label = LABELS[pred]
    print("Predicted label --> {}".format(label))

    if label == 'div':
        fixed_label = '/'
    elif label == 'equal':
        fixed_label = '='
    elif label == 'minus':
        fixed_label = '-'
    elif label == 'mul':
        fixed_label = '*'
    elif label == 'plus':
        fixed_label = '+'
    else:
        fixed_label = label

    return fixed_label
