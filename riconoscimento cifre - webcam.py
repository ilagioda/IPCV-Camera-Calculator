import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random as rng
import sys

rng.seed(12345)


def grab_frame(cap):
    """
    Method to grab a frame from the camera
    :param cap: the VideoCapture object
    :return: the captured image
    """
    ret, frame = cap.read()
    return frame


def bgr_to_rgb(image):
    """
    Convert a BGR image into grayscale
    :param image: the BGR image
    :return: the same image but in grayscale
    """
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def bgr_to_hsv(image):
    """
    Convert a BGR image into hsv
    :param image: the BGR image
    :return: the same image but in hsv
    """
    return cv.cvtColor(image, cv.COLOR_BGR2HSV)


def bgr_to_gray(image):
    """
    Convert a BGR image into grayscale
    :param image: the BGR image
    :return: the same image but in grayscale
    """
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def is_inside_outside(new_rect, rectangles):
    # Handle the case of an empty rectangles list
    if not rectangles:
        return 0
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

            # Check if rect is completely included in the new rectangle
            if min_x <= x0 and min_y <= y0 and x1 <= max_x and y1 <= max_y:
                rectangles.remove(rect)
                return 2

            # Check if the new rectangle is completely included in rect
            if x0 <= min_x and y0 <= min_y and max_x <= x1 and max_y <= y1:
                return 1

        return 0
    

def extract_element(image, start_point, end_point, num_elem):
    image = bgr_to_rgb(image)
    elem = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
    print("shape: {}x{} - start (min_x, min_y): {} {} - end (max_x, max_y): {} {}"
          .format(image.shape[0], image.shape[1], start_point[0], start_point[1], end_point[0], end_point[1]))
    name = "digits/elem" + str(num_elem + 1) + ".jpg"
    cv.imwrite(name, elem)


def try_blend_vertical(new_rect, rectangles):
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
                rectangles.remove(rett)
                new_rect = [min(x0, min_x), min(y0, min_y), max(x1, max_x), max(y1, max_y)]

        return new_rect


def detect_symbols(image):

    # Convert image to gray and blur it
    image_gray = bgr_to_gray(image)
    image_gray = cv.blur(image_gray, (3, 3))

    # Detect edges using Canny
    threshold = 30
    canny_output = cv.Canny(image_gray, threshold, threshold * 2)

    # Create the content of window
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    # Find contours
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Draw contours
    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv.drawContours(drawing, contours, i, color)

    # Find the convex hull and the rectangle for each contour
    hull_list = []
    rectangles = []
    for i in range(len(contours)):
        # Find the convex hull
        hull = cv.convexHull(contours[i])
        hull_list.append(hull)

        # Find the coordinates of the rectangle containing the letter
        min_x = sys.maxsize
        min_y = sys.maxsize
        max_x = -1
        max_y = -1
        for (num, vet) in enumerate(hull):
            # print("num = {}, coord = {}".format(num, vet))
            coord = vet[0]
            x = coord[0]
            y = coord[1]
            # print("x = {}, y = {}".format(x, y))
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
        #print("min_x = {}, min_y = {}, max_x = {}, max_y = {}".format(min_x, min_y, max_x, max_y))
        new_rect = [min_x, min_y, max_x, max_y]

        # Blend rectangles if they are vertically aligned
        # The rectangle is returned identical or merged with another one in the list (which is deleted)
        new_rect = try_blend_vertical(new_rect, rectangles)

        '''
        Controllo se il rettangolo appena trovato è incluso in un altro rettangolo oppure se un altro rettangolo è
        incluso in esso
        esito = 0 --> tutto ok, i due rettangoli non si includono a vicenda
                  --> viene mantenuto il rettangolo appena identificato
        esito = 1 --> il rettangolo appena identificato è incluso in un altro rettangolo più grande già identificato
                  --> viene scartato il rettangolo appena identificato
        esito = 2 --> il rettangolo appena identificato comprende un rettangolo più piccolo già identificato
                  --> viene mantenuto il rettangolo appena identificato (è stato eliminato il precedente)
        '''
        esito = is_inside_outside(new_rect, rectangles)
        # print("esito: {}".format(esito))
        if esito == 0 or esito == 2:
            rectangles.append(new_rect)

    # Process the final rectangle list
    num_elem = 0
    for rett in rectangles:
        min_x = rett[0]
        min_y = rett[1]
        max_x = rett[2]
        max_y = rett[3]
        # Start coordinate, represents the top left corner of rectangle
        start_point = (min_x, min_y)
        # Ending coordinate, represents the bottom right corner of rectangle
        end_point = (max_x, max_y)
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        # Draw the rectangle around the letter
        drawing = cv.rectangle(drawing, start_point, end_point, color, thickness)
        # Crop the element from the image
        extract_element(image, start_point, end_point, num_elem)
        num_elem += 1

    # Show everything in a window
    cv.imshow('Rettangoli attorno alle cifre', drawing)


def main():
    # Init the camera ---> 1 = WEBCAM ESTERNA!!!!
    cap = cv.VideoCapture(1)

    # Enable Matplotlib interactive mode
    plt.ion()

    # Create a figure to be updated
    fig = plt.figure()
    fig.canvas.mpl_connect("close_event", lambda event: cap.release())

    # Prep a variable for the first run
    ax_img = None

    # Parametri miei
    cont = 0
    pause_time = 1/30       # pause: 30 frames per second
    stop_cont = 30

    while cap.isOpened():
        # Get the current frame
        frame = grab_frame(cap)

        # Print cont
        print("Cont: {}".format(cont))

        # Controllo da quanti frame non vedo un oggetto giallo
        if cont == stop_cont:
            #final = bgr_to_rgb(frame)
            #cv.imshow("Ultimo frame", final)
            detect_symbols(frame)

        if ax_img is None:
            # Convert the current (first) frame in hsv (NOTA: necessario hsv perchè la funzione cv.inRange accetta quella scala)
            img = bgr_to_hsv(frame)

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
                if (area > 800):
                    # Yellow object found
                    cont = 0
                    x, y, w, h = cv.boundingRect(contour)
                    frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 10)
            if flag == 0:
                # Yellow object not found
                cont += 1

            ax_img = plt.imshow(bgr_to_rgb(frame))
            plt.axis("off")  # hide axis, ticks, ...
            plt.title("Camera Capture")
            plt.show()

        else:
            # Convert the current frame in hsv (NOTA: necessario hsv perchè la funzione cv.inRange accetta quella scala)
            img = bgr_to_hsv(frame)

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

            ax_img.set_data(bgr_to_rgb(frame))
            fig.canvas.draw()
            plt.pause(pause_time)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
