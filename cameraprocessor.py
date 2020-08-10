"""
Computer Vision-based module for processing a multimedia source (image or video)
and detect in it all the potential symbols that make up an arithmetical expression
"""

import numpy as np
import cv2 as cv
import utils
import multimedia
import neuralnetwork as net
import calculator


def try_blend(current, rectangles):
    """
    Check one rectangle against all others to try and blend some together, either based
    on a (significant) overlap or due to a combination of vertical alignment and proximity
    :param current: the rectangle that is trying to be merged with the others
    :param rectangles: list of other available rectangles
    :return: the input rectangle unchanged if it wasn't blended with anything, or the result of the
            merging operation (the rectangles list is modified by deleting merged rectangles)
    """
    if rectangles:

        max_distance = 60   # 60px (max vertical distance for blending items together)
        factor = 0.25       # 25% (minimum significant overlapping)

        min_x = current[0]
        min_y = current[1]
        max_x = current[2]
        max_y = current[3]

        for rect in rectangles:
            x0 = rect[0]
            y0 = rect[1]
            x1 = rect[2]
            y1 = rect[3]

            # Compute the vertical distance between the 2 rectangles
            distance_y = min(abs(max_y - y0), abs(min_y - y1))

            # Check horizontal interval intersection (shared X coordinates)
            shared_x = min(x1, max_x) - max(x0, min_x) if x0 < max_x and x1 > min_x else 0

            # Calculate the area of the intersection between rect and new_rect
            dx = min(max_x, x1) - max(min_x, x0)
            dy = min(max_y, y1) - max(min_y, y0)
            intersection_area = dx * dy if (dx >= 0) and (dy >= 0) else 0

            # Rectangles are considered vertically aligned if there is proximity along Y axis
            # and the 2 rectangles share a significant portion of their X coordinates (25%)
            aligned_vertical = (shared_x > factor * min(max_x - min_x, x1 - x0)
                                and distance_y <= max_distance)

            # Two rectangles are considered overlapped if the area of their intersection
            # covers a significant portion of each rectangle's own area (25%)
            overlapped = intersection_area > factor * min((x1 - x0) * (y1 - y0),
                        (max_x - min_x) * (max_y - min_y))

            # If atleast one condition is satisfied, merge the 2 rectangles together
            if overlapped or aligned_vertical:
                min_x = min(x0, min_x)
                min_y = min(y0, min_y)
                max_x = max(x1, max_x)
                max_y = max(y1, max_y)
                current = [min_x, min_y, max_x, max_y]
                # Delete the old rectangle
                rectangles.remove(rect)

    return current


def clear_outliers(rectangles):
    """
    Employs a clustering technique to try and remove possible noise points from actual symbols
    :param rectangles: list of all detected symbols (including real ones and possible outliers)
    :return: the updated rectangles list, hopefully without outliers
    """
    if not rectangles:
        return []

    # Get height and center point coordinates for each rectangle
    heights = list(map(lambda r: r[3]-r[1], rectangles))
    centers = list(map(utils.rectangle_center, rectangles))

    # Remove min and max height values before computing the mean,
    # in order to avoid extreme points that may be outliers
    if len(heights) > 2:
        heights.remove(min(heights))
        heights.remove(max(heights))

    # Compute neighbourhood size based on average symbol height
    radius = np.uint(2.5 * (sum(heights) / len(heights)))
    threshold = 2

    # For each rectangle, count how many neighbours it has inside the calculated radius
    core_points = []
    for center in centers:
        cnt = 0
        for other_center in centers:
            if other_center != center:  # Avoid self
                if utils.point_distance(center, other_center) < radius:
                    cnt += 1

        # If it has enough neighbours, it is marked as a core point
        if cnt >= threshold:
            core_points.append(center)

    # For all the rectangles that have not been marked as core points
    i = 0
    for center in centers:
        if center not in core_points:
            is_border = False
            for other_center in centers:
                if other_center != center:  # Avoid self
                    # If it is close to atleast 1 core point, it is marked as a border point
                    if utils.point_distance(center, other_center) < radius:
                        is_border = True
                        break

            # Remaining non-border points are considered outliers and therefore removed
            if not is_border:
                del rectangles[i]
                i -= 1
        i += 1

    return rectangles


def detect_symbols(image):
    """
    Runs image processing algorithms in order to detect symbols (digits and operators)
    in the provided image, cropping each of them and returning them as a list
    :param image: the image on which the detection has to be run
    :return: a list of the detected symbols as cropped images (or None)
    """
    if image is None:
        return None

    # 7x7 kernel for morphological operations
    kernel = np.array(
        [[0, 1, 1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1, 0]],
        np.uint8)

    # Convert image to gray and apply pre-processing
    image_proc = utils.bgr_to_gray(image)

    # Apply Gaussian blur
    image_proc = cv.GaussianBlur(image_proc, (9, 9), 0)

    # Apply image thresholding to separate handwriting from the background
    image_proc = cv.adaptiveThreshold(image_proc, 255,
                                      cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY, 9, 3)

    # Apply opening operator on the background (equals closing the foreground)
    # in order to fill gaps in the contours and 'repair' small holes
    image_proc = cv.morphologyEx(image_proc, cv.MORPH_OPEN, kernel)

    # Detect edges using Canny
    image_proc = cv.Canny(image_proc, 30, 60)

    # Create a copy of the image to draw debug info
    img_debug = np.copy(image)

    # Find contours
    (contours, _) = cv.findContours(image_proc, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    old_rectangles = []
    rectangles = []

    # Find the coordinates of bounding rectangle for each symbol contour
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        rectangles.append([x, y, x + w, y + h])

    # (Iterate this procedure until convergence)
    while len(rectangles) != len(old_rectangles):
        old_rectangles = rectangles.copy()
        rectangles.clear()

        for rect in old_rectangles:
            # Merge rectangles that are heavily intersected (overlapped) or vertically aligned
            # The rectangle is returned identical or merged with another one (which is deleted)
            rect = try_blend(rect, rectangles)

            # Add the processed rectangle to the list
            rectangles.append(rect)

    # Clear outliers from the detected list of rectangles
    rectangles = clear_outliers(rectangles)

    # Sort the rectangles from left to right as they appear in the frame
    rectangles.sort(key=lambda r: r[0])

    # Process the final rectangle list
    symbols = []
    for rect in rectangles:
        # Start coordinate, represents the top left corner of rectangle
        start_point = (rect[0], rect[1])
        # Ending coordinate, represents the bottom right corner of rectangle
        end_point = (rect[2], rect[3])
        # Draw the rectangle around the letter (with blue color)
        img_debug = cv.rectangle(img_debug, start_point, end_point, (255, 0, 0), thickness=2)
        # Crop the element from the image
        elem = utils.crop(image, start_point, end_point)
        # Append all valid elements to those that have to be processed by the neural network
        # The network operates on RGB images, therefore a color space conversion is performed
        if elem is not None:
            symbols.append(utils.bgr_to_rgb(elem))

    # Output a visual representation of the detection results
    cv.imwrite("./detected_rectangles.jpg", img_debug)          # TODO: riga da rimuovere

    # Retrieve the coordinates of the '=' (assumed to be the last symbol)
    equal_coordinates = rectangles[-1] if rectangles else []

    # Return list of extracted symbols and the coordinates of the "equal"
    return symbols, equal_coordinates


def detect_action(frame):
    """
    Detects the presence of hand-coloured objects in the image, or
    any movement that might happen inside the frame, which are taken
    as indicators that something is still going on (e.g. handwriting)
    and therefore the program has to wait a little longer
    :param frame: the image on which the detection algorithm has to be run
    :return: a pair (frame, has_actions) with the modified frame and a boolean
            that is True if some object or movement has been detected
    """

    # Convert the current frame in HSV (note: needed by cv.inRange())
    img = utils.bgr_to_hsv(frame)

    # TODO: implement more sophisticated technique for detecting movement
    # Thresholding with the usage of a mask for detecting yellow objects
    lower_yellow = np.array([20, 110, 110])
    upper_yellow = np.array([40, 255, 255])
    mask = cv.inRange(img, lower_yellow, upper_yellow)

    # Find contours of yellow objects
    (contours, _) = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Check the size of the detected yellow objects
    if contours:
        contours.sort(key=lambda c: cv.contourArea(c), reverse=True)
        if cv.contourArea(contours[0]) > 100:           # Only look at the largest object
            # Yellow object found
            x, y, w, h = cv.boundingRect(contours[0])
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 10)
            return frame, True

    return frame, False


def write_status(frame, status, counter=0):
    """
    Adds a text on the image (top-left corner is the standard), describing
    the current application status with different colors and parameters
    :param frame: the BGR image on which the text has to be added
    :param status: a string describing the status {WAITING, ERROR, SUCCESS, FINISHED}
    :return: the frame with the status text on it (or None)
    """
    if frame is None:
        return None

    if status not in {'WAITING', 'ERROR', 'SUCCESS', 'FINISHED'}:
        raise ValueError("Status can only be one of {WAITING, ERROR, SUCCESS, FINISHED}")

    # Default text parameters
    font = cv.FONT_HERSHEY_DUPLEX
    scale = 1
    thickness = 2
    margin = 15

    # Write the 'STATUS: ' word in a grey colour
    (text_width, text_height), _ = cv.getTextSize('STATUS: ', font, scale, thickness)
    position = (0 + margin, text_height + margin)
    frame = cv.putText(frame, 'STATUS:', position, font, scale, (50, 50, 50), thickness, cv.LINE_AA)

    # Determine the position of the status name based on the size of the text
    (_, text_height), _ = cv.getTextSize(status, font, scale, thickness)
    position = (text_width + margin, text_height + margin)

    # Choose specific text parameters for the current status
    if status == 'WAITING':
        color = (50, 190, 230)      # Yellow
        if counter > 1:
            status = ('WAITING (%d)' % counter)
    elif status == 'ERROR':
        color = (0, 0, 250)         # Red
    else:   # SUCCESS or FINISHED
        color = (0, 240, 0)         # Green

    return cv.putText(frame, status, position, font, scale, color, thickness, cv.LINE_AA)


def write_result(frame, result, equal_coords):
    """
    Writes the result of the arithmetical operation on the image, in the form
    of OpenCV-rendered text, positioning it near the '=' sign of the expression
    :param frame: the BGR image on which the result has to be written
    :param result: a string representing the numerical result of the expression
    :param equal_coords: coordinates [x0, y0, x1, y1] of the '=' rectangle
    :return: the frame with the result text written on it (or None)
    """
    if frame is None:
        return None

    # Retrieve image height and width
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Define text rendering parameters for cv.putText()
    font = cv.FONT_HERSHEY_SIMPLEX
    scale = (3 * (equal_coords[3] - equal_coords[1])) / 25    # Equal sign of 25px -> font scale = 3
    color = (0, 0, 0)   # Black
    thickness = 4   # px
    spacing = 30    # px

    # Get the size of the text/result to be displayed
    (text_width, text_height), _ = cv.getTextSize(result, font, scale, thickness)

    # Compute available_width
    available_width = frame_width - (equal_coords[2] + spacing)

    # Check if the result string can be contained horizontally in the image
    if text_width < available_width:
        # Everything's ok => write the result on the right of the '='
        coord = (equal_coords[2] + spacing, equal_coords[3])
    else:
        # The result string does not fit on the right of the '='
        # Compute available_height
        available_height = frame_height - (equal_coords[3] + spacing)

        # Check if the result string can be contained vertically in the image
        if text_height < available_height:
            # Everything's ok => write the result below the '='
            coord = (frame_width - text_width, equal_coords[3] + spacing + text_height)
        else:
            # There's not enough space around the '=' to contain the result string
            # => write the result at the top right corner of the image
            coord = (frame_width - text_width, text_height)

    # Render text on the image (coordinates refer to the bottom left corner of the text area)
    frame = cv.putText(frame, result, coord, font, scale, color, thickness, cv.LINE_AA)

    cv.imwrite("./result.jpg", frame)         # TODO: riga da rimuovere
    return frame


def run(sourceType, path):
    """
    Runs the actual Camera Calculator script, processing the input media source
    (image, video or webcam) and solving any arithmetical expression that
    it is able to detect inside the frame
    :param sourceType: the type of input media {image, video, webcam}
    :param path: the path to the input media
    """

    # Initialize the InputMedia and MediaPlayer (output) objects
    source = multimedia.InputMedia(sourceType, path)
    output = multimedia.MediaPlayer(sourceType, source.framerate()).start()

    # Status variables
    status = 'WAITING'
    result = None
    counter = 30

    # Loop through the entire input media, unless the program has been terminated by the user
    while source.isOpened() and not output.stopped():

        # Get the current frame
        frame = source.read()
        if frame is None:
            status = 'FINISHED'
            break

        # Run handwriting detection for (live or recorded) video inputs
        if sourceType in ['video', 'webcam']:
            frame, has_actions = detect_action(frame)

            if has_actions:
                # Reset to 'WAITING' status
                status = 'WAITING'
                result = None
                counter = 30
            else:
                # No object detected, keep decrementing the counter
                counter -= 1

        # Check if it's time to run the detection algorithm on the current frame/image
        if counter == 0 or sourceType == 'image':
            (symbols, equal_coordinates) = detect_symbols(frame)

            if symbols:
                # Initialize the array for the predicted symbols
                predicted = []
                for s in symbols:
                    # Prepare the image (pre-processing)
                    prepared_symbol = net.prepare_image(s)

                    # Predict the class label using a neural network
                    predicted_symbol = net.predict_symbol(prepared_symbol)

                    # Build the math expression by appending the prediction to the array of symbols
                    predicted.append(predicted_symbol)

                # Do the computation (and catch all possible errors)
                try:
                    (status, value) = calculator.compute(predicted)
                except Exception as err:
                    status = 'ERROR'
                    value = str(err)

                # Show the outcome to the user
                if status == 'SUCCESS':
                    expression_str = "".join(predicted)
                    result = utils.float_to_str(value)
                    # Print the result in the console
                    print(status + ": " + expression_str + result)

                elif status == 'ERROR':
                    print(status + ": " + value)

        if status != 'ERROR' and result is not None:
            # If available, print the result on the frame
            frame = write_result(frame, result, equal_coordinates)

        # Show the processed frame to the user
        frame = write_status(frame, status, counter)
        output.show(frame)

        # When working on an image, the program stops after the first iteration
        if sourceType == 'image':
            break

    # Close the MediaPlayer output (waiting for its termination)
    output.signal_end()
    output.close()
