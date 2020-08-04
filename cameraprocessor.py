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


def extract_element(image, start_point, end_point):
    """
    Extract an element from the frame, by specifying its coordinates
    :param image: the source image
    :param start_point: the top-left point of the crop area
    :param end_point: the bottom-right point of the crop area
    :return: the cropped portion of the image (in RGB)
    """
    if image is None:
        return None

    elem = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
    elem = utils.bgr_to_rgb(elem)
    return elem


def try_blend_intersected(new_rect, rectangles):
    """
    TODO: write docstring
    """
    # Handle the case of an empty rectangles list
    if not rectangles:
        return new_rect
    else:
        min_x = new_rect[0]
        min_y = new_rect[1]
        max_x = new_rect[2]
        max_y = new_rect[3]

        for rect in rectangles:
            x0 = rect[0]
            y0 = rect[1]
            x1 = rect[2]
            y1 = rect[3]

            # Calculate the area of the intersection between rect and new_rect
            intersection_area = 0
            dx = min(max_x, x1) - max(min_x, x0)
            dy = min(max_y, y1) - max(min_y, y0)
            if (dx >= 0) and (dy >= 0):
                intersection_area = dx * dy

            factor = 0.25  # 25%

            # If the 2 rectangles are heavily overlapped, merge them together
            area_rect = (x1 - x0) * (y1 - y0)
            area_new_rect = (max_x - min_x) * (max_y - min_y)
            if intersection_area >= factor * min(area_rect, area_new_rect):
                rectangles.remove(rect)
                new_rect = [min(x0, min_x), min(y0, min_y), max(x1, max_x), max(y1, max_y)]
                # The rectangle has changed and therefore update the coordinates
                min_x = new_rect[0]
                min_y = new_rect[1]
                max_x = new_rect[2]
                max_y = new_rect[3]

        return new_rect


def try_blend_vertical(new_rect, rectangles):
    """
    TODO: write docstring
    """
    # Handle the case of an empty rectangles list
    if not rectangles:
        return new_rect
    else:
        min_x = new_rect[0]
        min_y = new_rect[1]
        max_x = new_rect[2]
        max_y = new_rect[3]

        for rect in rectangles:
            x0 = rect[0]
            y0 = rect[1]
            x1 = rect[2]
            y1 = rect[3]

            # If rect is vertically aligned (but separated) w.r.t. the new rectangle
            if (x0 < max_x and x1 > min_x) and (y0 > max_y or y1 < min_y):
                # Remove rect from the list and return a new rectangle obtained by merging the 2
                rectangles.remove(rect)
                new_rect = [min(x0, min_x), min(y0, min_y), max(x1, max_x), max(y1, max_y)]
                # The rectangle has changed and therefore update the coordinates
                min_x = new_rect[0]
                min_y = new_rect[1]
                max_x = new_rect[2]
                max_y = new_rect[3]

        return new_rect


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
            # Merge together rectangles that are heavily intersected (overlapped)
            # The rectangle is returned identical or merged with another one in the list (which is deleted)
            rect = try_blend_intersected(rect, rectangles)

            # Blend rectangles if they are vertically aligned
            # The rectangle is returned identical or merged with another one in the list (which is deleted)
            rect = try_blend_vertical(rect, rectangles)

            # Add the processed rectangle to the list
            rectangles.append(rect)

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
        elem = extract_element(image, start_point, end_point)
        # Append all valid elements to those that have to be processed by the neural network
        if elem is not None:
            symbols.append(elem)

    # Show everything in a window
    #cv.imshow('Detection results', drawing)        # TODO: commentato solo perchè Colab è stupido

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
    fontScale = 3
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

    # cv.imwrite("video/result.jpg", img_with_text)         # TODO: riga da rimuovere
    print("HO STAMPATO A VIDEO IL RISULTATO!!")             # TODO: riga da rimuovere

    # Displaying the image
    plt.imshow(img_with_text)
    plt.title("Operazione matematica")
    plt.show()


def main():

    # Initialize the random number generator
    random.seed(12345)

    # Init the camera
    #cap = cv.VideoCapture(1) ---> 1 = WEBCAM ESTERNA!!!!
    cap = cv.VideoCapture("video/matita 74+80.mp4")

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
