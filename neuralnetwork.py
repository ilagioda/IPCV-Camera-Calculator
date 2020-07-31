import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import utils

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import alexnet

from PIL import Image

NUM_CLASSES = 16
NN_PATH = './preproc_augm_symbols_net.pth'
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'div', 'equal', 'minus', 'mul', 'plus', 'separator']

def prepare_image(image):
    """
    Image pre-processing, before running them through the neural network
    :param image: an RGB image of a symbol that has to be processed by the neural network
    :return: preparred image
    """
    if image is None:
        return

    # Convert to grayscale
    img_gray = utils.rgb_to_gray(image)

    # Apply thresholding to the image, separating the background from the symbol
    thresh = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 2)

    # Define a kernel for morphological operations
    kernel = np.array(
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]],
        np.uint8)

    # Apply closing operator to remove noise
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

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
    image_prepared[y_start:y_end, x_start:x_end] = thresh

    return image_prepared


def predict_symbol(img_cv):
    """
    Given a certain image containing a symbol, allows to predict the class label, i.e.
    the category to which the symbol belongs, through the usage of a pre-trained neural network (AlexNet)
    :param img: an image containing a symbol
    :return: predicted label, i.e. predicted symbol
    """

    # Convert into PIL format
    img_cv = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)
    img = Image.fromarray(img_cv).convert("RGB")

    # Define transforms for the prediction phase
    eval_transform = transforms.Compose([transforms.Resize(224),                                 # Resizes short size of the PIL image to 256
                                         transforms.ToTensor(),                                  # Turn PIL Image to torch.Tensor
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizes tensor with mean and standard deviation
                                         ])

    # Define a new model
    net = alexnet()

    # Change the last layer of AlexNet to output 16 classes
    net.classifier[6] = nn.Linear(4096, NUM_CLASSES)

    # Load the already-trained model
    net.load_state_dict(torch.load(NN_PATH))

    # Set the evaluation mode
    net.eval()

    # Prepare the image for the network
    img = eval_transform(img).float()
    img = img.unsqueeze(0)

    # Forward the image through the network
    outputs = net(img)

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
    elif label == 'separator':
        fixed_label = '.'
    else:
        fixed_label = label

    return fixed_label