import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random as rng

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


def bgr_to_gray(image):
    """
    Convert a BGR image into grayscale
    :param image: the BGR image
    :return: the same image but in grayscale
    """
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def main():
    # init the camera ---> 1 = WEBCAM ESTERNA!!!!
    cap = cv.VideoCapture(1)

    # enable Matplotlib interactive mode
    plt.ion()

    # create a figure to be updated
    fig = plt.figure()
    fig.canvas.mpl_connect("close_event", lambda event: handle_close(event, cap))

    # prep a variable for the first run
    ax_img = None

    threshold = 70
    rng.seed(12345)

    while cap.isOpened():
        # get the current frame
        frame = grab_frame(cap)
        if ax_img is None:
            # convert the current (first) frame in grayscale
            img = bgr_to_gray(frame)

            # Detect edges using Canny
            canny_output = cv.Canny(img, threshold, threshold * 2)

            # Find contours
            contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            # Find the convex hull object for each contour
            hull_list = []
            for i in range(len(contours)):
                hull = cv.convexHull(contours[i])
                hull_list.append(hull)

            # Draw contours + hull results
            drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
            for i in range(len(contours)):
                #color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
                color = (255, 255, 255)
                cv.drawContours(drawing, contours, i, color)
                cv.drawContours(drawing, hull_list, i, color)
                # Show in a another window
                # cv.imshow('Contours', drawing)

            ax_img = plt.imshow(drawing , "gray")
            plt.axis("off")  # hide axis, ticks, ...
            plt.title("Camera Capture")
            # show the plot!
            plt.show()
        else:
            # convert the current frame in grayscale
            img = bgr_to_gray(frame)

            # Detect edges using Canny
            canny_output = cv.Canny(img, threshold, threshold * 2)

            # Find contours
            contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            # Find the convex hull object for each contour
            hull_list = []
            for i in range(len(contours)):
                hull = cv.convexHull(contours[i])
                hull_list.append(hull)

            # Draw contours + hull results
            drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
            for i in range(len(contours)):
                #color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
                color = (255, 255, 255)
                cv.drawContours(drawing, contours, i, color)
                cv.drawContours(drawing, hull_list, i, color)
                # Show in a another window
                #cv.imshow('Contours', drawing)

            # set the current frame as the data to show
            ax_img.set_data(drawing)
            # update the figure associated to the shown plot
            fig.canvas.draw()
            plt.pause(1/30)  # pause: 30 frames per second


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)