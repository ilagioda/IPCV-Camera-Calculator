import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

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

    # Parametri miei
    cont = 0
    pause_time = 1/30       # pause: 30 frames per second
    stop_cont = 30

    while cap.isOpened():
        # get the current frame
        frame = grab_frame(cap)

        print("Cont: {}".format(cont))

        # Controllo da quanti frame non vedo un oggetto rosso
        if cont == stop_cont:
            final = bgr_to_rgb(frame)
            cv.imshow("Ultimo frame", final)

        if ax_img is None:
            # Convert the current (first) frame in hsv (NOTA: necessario hsv perchè la funzione cv.inRange accetta quella scala)
            img = bgr_to_hsv(frame)

            # Maschera per rilevare il COLORE GIALLO
            lower_yellow = np.array([20, 110, 110])
            upper_yellow = np.array([40, 255, 255])
            mask = cv.inRange(img, lower_yellow, upper_yellow)

            # Maschera per rilevare il COLORE ROSSO (prove)
            # Gen lower mask (0-5) and upper mask (175-180) of RED
            #mask1 = cv.inRange(img, (0, 50, 20), (5, 255, 255))
            #mask2 = cv.inRange(img, (175, 50, 20), (180, 255, 255))
            # Merge the mask and crop the red regions
            #mask = cv.bitwise_or(mask1, mask2)

            # Find contours of red objects
            (contours, _) = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            flag = 0
            for contour in contours:
                flag = 1
                area = cv.contourArea(contour)
                if (area > 800):
                    # Ho rilevato un oggetto rosso
                    cont = 0
                    x, y, w, h = cv.boundingRect(contour)
                    frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 10)
            if flag == 0:
                # Non ho rilevato un oggetto rosso
                cont += 1

            ax_img = plt.imshow(bgr_to_rgb(frame))
            plt.axis("off")  # hide axis, ticks, ...
            plt.title("Camera Capture")
            # show the plot!
            plt.show()
        else:
            # Convert the current frame in hsv (NOTA: necessario hsv perchè la funzione cv.inRange accetta quella scala)
            img = bgr_to_hsv(frame)

            # Maschera per rilevare il COLORE GIALLO
            lower_yellow = np.array([20, 110, 110])
            upper_yellow = np.array([40, 255, 255])
            mask = cv.inRange(img, lower_yellow, upper_yellow)

            # Maschera per rilevare il COLORE ROSSO (prove)
            # Gen lower mask (0-5) and upper mask (175-180) of RED
            #mask1 = cv.inRange(img, (0, 50, 20), (5, 255, 255))
            #mask2 = cv.inRange(img, (175, 50, 20), (180, 255, 255))
            # Merge the mask and crop the red regions
            #mask = cv.bitwise_or(mask1, mask2)

            # Find contours of red objects
            (contours, _) = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            flag = 0
            for contour in contours:
                flag = 1
                area = cv.contourArea(contour)
                if (area > 800):
                    # Ho rilevato un oggetto rosso
                    cont = 0
                    x, y, w, h = cv.boundingRect(contour)
                    frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 10)
            if flag == 0:
                # Non ho rilevato un oggetto rosso
                cont += 1

            # set the current frame as the data to show
            ax_img.set_data(bgr_to_rgb(frame))
            # update the figure associated to the shown plot
            fig.canvas.draw()
            plt.pause(pause_time)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)