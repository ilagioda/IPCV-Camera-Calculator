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


def handle_close(event, cap):
    """
    Handle the close event of the Matplotlib window by closing the camera capture
    :param event: the close event
    :param cap: the VideoCapture object to be closed
    """
    cap.release()


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


def is_inside_outside(rettangoli, min_x, min_y, max_x, max_y):
    # Controllo se la lista rettangoli è vuota
    if not rettangoli:
        rettangoli.append([min_x, min_y, max_x, max_y])
        return 0
    else:
        for rett in rettangoli:
            x0 = rett[0]
            y0 = rett[1]
            x1 = rett[2]
            y1 = rett[3]
            #print("Rett... x0 = {}, y0 = {}, x1 = {}, y1 = {}".format(x0, y0, x1, y1))

            # Controllo se rett è incluso nel nuovo rettangolo
            if min_x <= x0 and min_y <= y0 and x1 <= max_x and y1 <= max_y:
                rettangoli.append([min_x, min_y, max_x, max_y])
                rettangoli.remove(rett)
                return 2

            # Controllo se il nuovo rettangolo è incluso in rett
            if x0 <= min_x and y0 <= min_y and max_x <= x1 and max_y <= y1:
                return 1
            if x0 == min_x and y0 == min_y and max_x == x1 and max_y == y1:
                return 1

        rettangoli.append([min_x, min_y, max_x, max_y])
        return 0

def estrai_elemento(image, start_point, end_point, num_elem):
    num_elem += 1
    image = bgr_to_rgb(image)
    elem = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
    print("shape: {}x{} - start (min_x, min_y): {} {} - end (max_x, max_y): {} {}".format(image.shape[0], image.shape[1], start_point[0], start_point[1], end_point[0], end_point[1]))
    nome = "digits/elem" + str(num_elem) + ".jpg"
    cv.imwrite(nome, elem)


def riconosci_cifre(image):
    num_elem = 0
    # Convert image to gray and blur it
    image_gray = bgr_to_gray(image)
    image_gray = cv.blur(image_gray, (3, 3))

    # Build a vector to maintain the already-found-rectangles' dimensions
    rettangoli = []

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

        '''
        Controllo se il rettangolo appena trovato è incluso in un altro rettangolo oppure se un altro rettangolo è
        incluso in esso
        esito = 0 --> tutto ok, i due rettangoli non si includono a vicenda
                  --> stampo a video tale rettangolo appena identificato
        esito = 1 --> il rettangolo appena identificato è incluso in un altro rettangolo più grande già identificato
                  --> non stampo a video tale rettangolo appena identificato
        esito = 2 --> il rettangolo appena identificato comprende un rettangolo più piccolo già identificato
                  --> stampo a video tale rettangolo appena identificato
        '''
        esito = is_inside_outside(rettangoli, min_x, min_y, max_x, max_y)
        # print("esito: {}".format(esito))
        if esito == 0 or esito == 2:
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
            estrai_elemento(image, start_point, end_point, num_elem)
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
    fig.canvas.mpl_connect("close_event", lambda event: handle_close(event, cap))

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
            riconosci_cifre(frame)

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
                if (area > 800):
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