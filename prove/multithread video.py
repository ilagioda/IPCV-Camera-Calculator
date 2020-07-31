from threading import Thread
from queue import Queue
import random
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2


class VideoStreamReader:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread and accumulates them in a queue.
    """

    def __init__(self, srctype, path, queueSize=1):
        """
        Initialize the VideoCapture stream and the frame buffer
        """
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.queue = Queue(maxsize=queueSize)

        if srctype == 'webcam':
            # Webcam
            self.rate = 30
        elif srctype == 'video':
            # Video file
            self.rate = self.stream.get(cv2.CAP_PROP_FPS)

    def start(self):
        """
        Start a thread to read frames from the video stream
        """
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        """
        Loop until the thread is stopped
        """
        while not self.stopped:

            # If the queue is not full...
            if not self.queue.full():
                # ...read the next frame from the video stream
                (grabbed, frame) = self.stream.read()

                # Check if we have reached the end of the video file
                # or if the webcam has been disconnected
                if not grabbed:
                    self.stop()
                    return

                # Add the new frame to the queue
                self.queue.put(frame)

            else:
                # Avoid useless looping, pause for the time needed to consume 1 frame
                time.sleep(1/self.rate)

    def stop(self):
        """
        Sets the VideoStreamReader `stopped` flag
        """
        self.stopped = True

    def read(self):
        """
        Get the next frame in the queue (blocking operation if queue is empty)
        """
        return self.queue.get()
        #(grabbed, frame) = self.stream.read()
        #if not grabbed:
        #    self.stop()
        #return frame

    def finished(self):
        """
        Returns false as long as there are frames that need to be processed
        """
        return self.stopped and self.queue.qsize() == 0


def thread_video_test():
    """
    Dedicated thread for grabbing video frames with VideoStreamReader object.
    Main thread shows video frames.
    """

    # Initialize the random number generator
    random.seed(12345)

    video_getter = VideoStreamReader('video', "C:\\Users\\JackaL\\source\\repos\\IPCV-Camera-Calculator\\video\\matita 17+65.mp4").start()

    # Set up MatPlotLib plot to draw frames in interactive mode
    plt.ion()

    # Create a figure to be updated
    fig = plt.figure()
    fig.canvas.mpl_connect("close_event", lambda event: video_getter.stop())
    ax_img = None

    # Timer parameters
    cont = 0
    stop_cont = 30

    t0 = time.time()

    while not video_getter.finished():

        frame = video_getter.read()
        if frame is not None:
            height = frame.shape[0]
            width = frame.shape[1]

            # Resize the frame to 960x540 (preserve aspect ratio)
            factor = min(960 / width, 540 / height)
            new_size = (int(width * factor), int(height * factor))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_CUBIC)

            # Print cont
            print("Cont: {}".format(cont))

            # Check if the tracked yellow object has exited the video for enough frames
            if cont == stop_cont:
                print("Done")

            # Convert the current frame in HSV (note: needed by cv.inRange())
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Thresholding with the usage of a mask for detecting the yellow
            lower_yellow = np.array([20, 110, 110])
            upper_yellow = np.array([40, 255, 255])
            mask = cv2.inRange(img, lower_yellow, upper_yellow)

            # Find contours of yellow objects
            (contours, _) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            flag = 0
            for contour in contours:
                flag = 1
                area = cv2.contourArea(contour)
                if area > 800:
                    # Yellow object found
                    cont = 0
                    x, y, w, h = cv2.boundingRect(contour)
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 10)
            if flag == 0:
                # Yellow object not found
                cont += 1

            if ax_img is None:
                ax_img = plt.imshow(frame)
                plt.axis("off")  # hide axis, ticks, ...
                plt.title("Buffered VideoCapture")
                plt.show()
            else:
                ax_img.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                fig.canvas.draw()

        else:
            break
    
    t1 = time.time() - t0
    print("Time elapsed: ", t1)


if __name__ == "__main__":
    try:
        thread_video_test()
    except KeyboardInterrupt:
        sys.exit(0)
    