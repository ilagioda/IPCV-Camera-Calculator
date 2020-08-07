import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random
import utils
import neuralnetwork as net
import calculator

def grab_frame(cap):
    """
    Method to grab a frame from the camera
    :param cap: the VideoCapture object
    :return: the captured image
    """
    ret, frame = cap.read()

    if frame is not None:
        height = frame.shape[0]
        width = frame.shape[1]

        # Resize the frame to 960x540 (preserve aspect ratio)
        factor = min(960 / width, 540 / height)
        new_size = (int(width * factor), int(height * factor))
        frame = cv.resize(frame, new_size)

    return frame


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


def clear_outliers(rectangles, image):
    """
    Employs a clustering technique to try and remove possible noise points from actual symbols
    :param rectangles: list of all detected symbols (including real ones and possible outliers)
    :return: the updated rectangles list, hopefully without outliers
    """
    if not rectangles or len(rectangles) < 4:
        return rectangles

    # Get height and center point coordinates for each rectangle
    heights = list(map(lambda r: r[3]-r[1], rectangles))
    centers = list(map(utils.rectangle_center, rectangles))

    # Remove min and max height values before computing the mean,
    # in order to avoid extreme points that may be outliers
    heights.remove(min(heights))
    heights.remove(max(heights))

    # Compute neighbourhood size based on average symbol height
    radius = np.uint(2 * (sum(heights) / len(heights)))
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
            cv.circle(image, center, radius, (0, 255, 0))      # Debug
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
                        cv.circle(image, center, radius, (255, 0, 0))      # Debug
                        break

            # Remaining non-border points are considered outliers and therefore removed
            if not is_border:
                cv.circle(image, center, radius, (0, 0, 255))      # Debug
                del rectangles[i]
                i -= 1
        i += 1

    cv.imwrite("./circles.jpg", image)         # TODO: riga da rimuovere

    return rectangles


def detect_symbols(image):
    """
    TODO: write docstring
    """
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
    image_gray = utils.bgr_to_gray(image)

    # Apply blur
    image_gray = cv.GaussianBlur(image_gray, (9, 9), 0)

    # Apply thresholding
    image_thresh = cv.adaptiveThreshold(image_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 9, 3)

    # Apply opening operators
    image_thresh = cv.morphologyEx(image_thresh, cv.MORPH_OPEN, kernel)

    # Debug
    #cv.imshow('Thresholded image', image_thresh)

    # Detect edges using Canny
    threshold = 30
    canny_output = cv.Canny(image_thresh, threshold, threshold * 2)

    # Create the content of window
    drawing = np.copy(image)

    # Find contours
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Draw contours
    # for i in range(len(contours)):
    #     color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
    #     cv.drawContours(drawing, contours, i, color)

    old_rectangles = []
    rectangles = []

    # Find the coordinates of bounding rectangle for each symbol contour
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        rectangles.append([x, y, x + w, y + h])

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
    rectangles = clear_outliers(rectangles, np.copy(image))

    # Sort the rectangles from left to right as they appear in the frame
    rectangles.sort(key=lambda r: r[0])

    # Process the final rectangle list
    symbols = []
    for rect in rectangles:
        # Start coordinate, represents the top left corner of rectangle
        start_point = (rect[0], rect[1])
        # Ending coordinate, represents the bottom right corner of rectangle
        end_point = (rect[2], rect[3])
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        # Draw the rectangle around the letter
        drawing = cv.rectangle(drawing, start_point, end_point, color, thickness)
        # Crop the element from the image
        elem = utils.crop(image, start_point, end_point)
        # Append all valid elements to those that have to be processed by the neural network
        # The network operates on RGB images, therefore a color space conversion is performed
        if elem is not None:
            symbols.append(utils.bgr_to_rgb(elem))

    # Show everything in a window
    cv.imwrite("./detected_rectangles.jpg", drawing)         # TODO: riga da rimuovere

    # Retrieve the coordinates of the "equal"
    if rectangles:
        equal_coordinates = rectangles[-1]
    else:
        equal_coordinates = []

    # Return list of extracted symbols and the coordinates of the "equal"
    return symbols, equal_coordinates


def displayResult(img, result, equal_coordinates):
    # Retrieve image height and width
    img_height = img.shape[0]
    img_width = img.shape[1]

    # Define parameters for the cv.putText
    # ... font
    font = cv.FONT_HERSHEY_SIMPLEX
    # ... fontScale
    #fontScale = 3
    fontScale = (3 * (equal_coordinates[3] - equal_coordinates[1])) / 25        # NOTA: 25 = altezza dell'"uguale" che ho quando le cifre
                                                                                # del risultato mi sembrano ben
                                                                                # proporzionate con fontScale = 3
    # ... color (black)
    color = (0, 0, 0)
    # ... line thickness (in px)
    fontThickness = 4

    # Define how much space we want between the equal and the result (in px)
    result_space = 30

    # Get the size of the text/result to be displayed
    (result_width, result_height), baseline = cv.getTextSize(result, font, fontScale, fontThickness)

    # Compute available_width
    # REMIND: equal_coordinates is an array in the form [x, y, x + w, y + h]
    available_width = img_width - (equal_coordinates[2] + result_space)

    # Check if the string "result" can be contained horizontally in the image
    if result_width < available_width:
        # Everything ok => I can write the result horizontally, after the 'equal'
        # Define the coordinates
        coord = (equal_coordinates[2] + result_space, equal_coordinates[3])
    else:
        # The string "result" can not be contained horizontally in the image
        # Compute available_height
        available_height = img_height - (equal_coordinates[3] + result_space)

        # Check if the string "result" can be contained vertically in the image
        if result_height < available_height:
            # Everything ok => I can write the result vertically, under the 'equal'
            coord = (equal_coordinates[0], equal_coordinates[3] + result_space + result_height)
        else:
            # The string "result" can not be contained neither horizontally nor vertically in the image
            # => I write the result in the top left corner of the image
            coord = (0, result_height)

    # Using cv2.putText() method
    # NOTA: coord è il vertice in basso a SX della stringa da posizionare (origine = angolo in alto a SX)!
    img_with_text = cv.putText(img, result, coord, font, fontScale, color, fontThickness, cv.LINE_AA)

    cv.imwrite("./result.jpg", img_with_text)         # TODO: riga da rimuovere

    # Displaying the image
    plt.imshow(img_with_text)
    plt.title("Operazione matematica")
    plt.show()


def main():

    # Initialize the random number generator
    random.seed(12345)

    # Init the camera
    #cap = cv.VideoCapture(1) ---> 1 = WEBCAM ESTERNA!!!!
    cap = cv.VideoCapture("video/16+40.mp4")

    # Enable Matplotlib interactive mode
    plt.ion()

    # Create a figure to be updated
    fig = plt.figure()
    fig.canvas.mpl_connect("close_event", lambda event: cap.release())

    # Prep a variable for the first run
    ax_img = None

    # Timer parameters
    cont = 0
    pause_time = 1 / 30  # pause: 30 frames per second
    stop_cont = 30

    # Initialize the array that will contain the predicted symbols
    predicted = []

    while cap.isOpened():
        # Get the current frame
        frame = grab_frame(cap)
        if frame is None:
            break

        # Print cont
        print("Cont: {}".format(cont))

        # Check if the tracked yellow object has exited the video for enough frames
        if cont == stop_cont:
            # Run the detection algorithm on the current frame
            symbols, equal_coordinates = detect_symbols(frame)

            if symbols:
                for s in symbols:
                    # Prepare the image (pre-processing)
                    prepared_symbol = net.prepare_image(s)

                    # Predict the class label using a neural network
                    predicted_symbol = net.predict_symbol(prepared_symbol)

                    # Build the math expression by appending the prediction to the array of symbols
                    predicted.append(predicted_symbol)

                # Do the computation
                (outcome, value) = calculator.compute(predicted)

                # Show 'result' to the user
                print(outcome)
                if outcome == 'SUCCESS':
                    expression_str = ""
                    for symbol in predicted:
                        expression_str += symbol

                    # Print the result in the console
                    print(expression_str + utils.float_to_str(value))

                    # Show the result on the screen
                    displayResult(frame, utils.float_to_str(value), equal_coordinates)

                elif outcome == 'ERROR':
                    print("Reason: " + value)

                # End the "cap.isOpened" while
                break

        # Convert the current frame in HSV (note: needed by cv.inRange())
        img = utils.bgr_to_hsv(frame)

        # Thresholding with the usage of a mask for detecting the yellow
        lower_yellow = np.array([20, 110, 110])
        upper_yellow = np.array([40, 255, 255])
        mask = cv.inRange(img, lower_yellow, upper_yellow)

        # Find contours of yellow objects
        (contours, _) = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        flag = 0
        for contour in contours:
            flag = 1
            area = cv.contourArea(contour)
            if area > 800:
                # Yellow object found
                cont = 0
                x, y, w, h = cv.boundingRect(contour)
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 10)
        if flag == 0:
            # Yellow object not found
            cont += 1

        # Draw the frame in the viewport
        if ax_img is None:
            ax_img = plt.imshow(utils.bgr_to_rgb(frame))
            plt.axis("off")  # hide axis, ticks, ...
            plt.title("Camera Capture")
            plt.show()
        else:
            ax_img.set_data(utils.bgr_to_rgb(frame))
            fig.canvas.draw()
            plt.pause(pause_time)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
