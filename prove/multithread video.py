from threading import Thread
import queue
import sys
import time
import numpy as np
import cv2


class VideoPlayer:
    """
    Class that shows frames from a video sequence through cv.imshow()
    with a dedicated thread, by picking them from a queue.
    """

    def __init__(self, framerate=30, queueSize=30):
        """
        Initialization
        """
        self.stopped = False
        self.rate = framerate
        self.queue = queue.Queue(maxsize=queueSize)


    def start(self):
        """
        Start a thread to show frames from the video buffer
        """
        t = Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self


    def update(self):
        """
        Loop until the thread is stopped
        """
        while not self.stopped:

            # If the queue is not empty
            if not self.queue.empty():
                # Get the next frame from the video buffer
                frame = self.queue.get()

                # And show it on the screen
                cv2.imshow('Video', frame)

                # Wait for the proper time (int(1000/self.rate))
                # and also check if the user has pressed the ESC key to quit
                if cv2.waitKey(int(1000/self.rate)) & 0xFF == ord('\x1b'):
                    self.stopped = True
                    break

                # Check if the window has been closed by the user
                if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
                    self.stopped = True
                    break


    def show(self, frame):
        """
        Push the next frame in the queue (blocking operation if queue is full)
        """
        done = False
        while not done and not self.stopped:
            try:
                self.queue.put(frame, timeout=1)
                done = True
            except queue.Full:
                done = False


    def finished(self):
        """
        Returns false as long as the video player is running
        """
        return self.stopped


def grab_frame(cap):
    """
    Method to grab a frame from the camera
    :param cap: the VideoCapture object
    :return: the captured image
    """
    _, frame = cap.read()

    if frame is not None:
        height = frame.shape[0]
        width = frame.shape[1]

        # Resize the frame to 960x540 (preserve aspect ratio)
        factor = min(960 / width, 540 / height)
        new_size = (int(width * factor), int(height * factor))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_CUBIC)

    return frame


if __name__ == "__main__":

    # Initialize video objects
    cap = cv2.VideoCapture("E:\\DOWNLOAD\\16+40.mp4")
    video_shower = VideoPlayer().start()

    # Timer parameters
    cont = 0
    stop_cont = 30

    t0 = time.time()

    while cap.isOpened():

        # Get the current frame
        frame = grab_frame(cap)
        if frame is None:
            break

        if video_shower.finished():
            cap.release()
            break

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

        # Enqueue the frame for showing it on the screen
        video_shower.show(frame)

    t1 = time.time() - t0
    print("Time elapsed: ", t1)
    