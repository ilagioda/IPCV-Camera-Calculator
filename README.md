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
4. [Further development](#Further-development)

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

The following paragraphs provide brief descriptions of the main operations that are carried on by Camera Calculator during its execution.
For more details, it is recommended to look into the code itself, which is documented pretty well and therefore should provide the best source of information regarding how each feature works.

<ins>NOTE</ins>: Camera Calculator has been developed assuming that the arithmetical expression would be hand-written on a white sheet of paper, and that only 1 expression would be visible at a time; outside of these assumptions, the application may not work as intended or at all.

### Hand detection (see `detect_hand()` in *cameraprocessor.py*)

This operation is performed only when the application is working on a video file or a webcam feed; the presence of an hand in the frame is used as an indicator that the user may still be writing out the expression and therefore the symbol detection should not run yet.

In practice, the application detects the presence of skin-coloured objects by applying a `cv.inRange()` operation to find all colours between (0, 48, 100) and (20, 255, 255) and then post-processing the results. Objects shapes are not taken into account, therefore anything with a pink-ish colour may create a false positive.

### Symbols detection (see `detect_symbols()` in *cameraprocessor.py*)

In order to identify symbols inside the frame, a blurred version of the image is run through the `cv.adaptiveThreshold()` function which is great at separating the foreground from a white background even under non-uniform lighting conditions; the Canny edge detection algorithm and `cv.findContours()` are then used to get the actual symbol shapes based on their contours.

A sequence of custom post-processing operations are run on the bounding boxes of the detected shapes (`cv.boundingRect()`), in order to blend together those that are overlapped or close enough to be considered part of the same symbol. This is necessary because edge detection results are not perfect and often split a single symbol into multiple objects due to gaps in the writing itself or suboptimal focus by the camera; with this technique we are able to recover the majority of these errors, but there may also be cases where distinct symbols get merged into a single object because they were written too close to each other.

Results of these procedure (bouding box coordinates) can be then used to crop portions of the image that contain a mathematical symbol each.

### Noise removal (see `clear_outliers()` in *cameraprocessor.py*)

Due to the large number of false symbol detections (caused by shadows, extraneous objects inside the frame or simply unwanted signs on the sheet of paper), a noise removal algorithm was required to clean the set of detected symbols from possible outliers.

A first cleaning pass, used to remove small noise points that may be generated by the thresholding and edge detection algorithms, is run by removing all candidates that are less than 15px wide or tall, which is too small to be significant in a 960x540 frame.

Right after that, a DBSCAN clustering-inspired algorithm is used to remove from the candidates set any object that is isolated or too far from other symbols (this process takes into account the average symbol size in order to determine whether 2 items are at a reasonable distance from each other or not). This algorithm has proved sufficiently good at removing noise points in the typical use cases.

### Neural network (see *neuralnetwork.py*)

// TODO

### Output (see `write_status()` and `write_result()` in *cameraprocessor.py*)

Camera Calculator provides two different outputs to the user, both written as overlay text in the application window using the `cv.putText()` function:
* The ***status*** indicator is always present and informs the user about the current application status; it can take the following values
  * WAITING: (*'video'* or *'webcam'* modes only) skin-coloured objects are present in the frame and the application is waiting for them to disappear
  * ERROR: some kind of error has been encountered during the symbol detection procedure or the computation of the final result
  * SUCCESS: the expression has been resolved and the result is being shown in the output window
* The actual ***result*** of the expression is presented to the user by writing its value to the right of the '=' sign, immediately below it or at the top-right corner of the image, depending on the available space.

# Further development

Most of the current limitations of Camera Calculator derive from the usage of a small dataset of not-so-great quality. By using a better and larger dataset to train our classifier we would be able to provide this improvements to the application:
* Increase prediction accuracy for currently supported symbols (digits and operators)
* Add parenthesis to the set of supported arithmetical symbols
* Add support for decimal values, by introducing the 'separator' token (. or ,)

The last point could require some additional work, due to possible issues that may arise in the shapes post-processing and noise removal algorithms when introducing a 'dot' separator symbol that is very small in size and typically written very close to other digits.
