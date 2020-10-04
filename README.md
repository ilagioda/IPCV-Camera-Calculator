# IPCV-Camera-Calculator

Camera Calculator was developed as an academic project for the Image Processing and Computer Visions course at PoliTO.

It is a computer-vision based application, written in Python and using the [OpenCV](https://opencv.org/) library, capable of identifying handwritten arithmetical expressions from images, videos or a live webcam feed and presenting results to the user after having processed the symbols (with the help of a neural network) and solved each expression.

# Index
1. [Usage](#Usage)
2. [Project overview](#Project-overview)
3. [Technical details](#Technical-details)
   - [Hand detection](#Hand-detection)
   - [Symbols detection](#Symbols-detection)
   - [Noise removal](#Noise-removal)
   - [Neural network](#Neural-network)
   - [Output](#Output)
4. [Conclusions](#Conclusions)

# Usage

// TODO

# Project overview

* ***dataset\\***
  * contains the set of images used to train the classifier, split into *train* (training set) and *eval* (evaluation set)
* ***net\\***
  * contains the training scripts for the neural network, along with the final trained model (*NN.pth* file)
* ***images\\*** and ***videos\\***
  * contain several image and video files that can be used as test scenarios for the application
* ***camera-calculator.py:*** exposes a command line interface to launch the application and define some execution parameters
* ***cameraprocessor.py:*** being the core of the application, this module is able to process a multimedia source (image or video) and extract from it all the mathematical symbols written on a sheet of paper
* ***calculator.py:*** implements the logic required to parse sequences of symbols and compute the result of math operations
* ***multimedia.py:*** defines the InputMedia and MediaPlayer classes, providing multimedia I/O facilities and high-performance video playback
* ***neuralnetwork.py:*** employs a NN-based classifier to predict arithmetical symbols (digits and operators) from cropped images
* ***utils.py:*** defines some support and utility functions used throughout the application code


# Technical details

// TODO

### Hand detection

// TODO

### Symbols detection

// TODO

### Noise removal

// TODO

### Neural network

// TODO

### Output

// TODO

# Conclusions

// TODO
