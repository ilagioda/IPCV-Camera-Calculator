"""
Module with facilities for dealing with different multimedia types (image, video, live webcam),
in both input and output, with high performance and stability in mind
"""

from threading import Thread
import queue
import sys
import cv2 as cv


class MediaPlayer:
    """
    The MediaPlayer class represents an object that is capable of displaying
    multimedia (image or video) information on the user's screen, in a stable
    and efficient way (using a separate thread and frame buffering for videos,
    while displaying images in the most lightweight way possible)
    """

    def __init__(self, mediaType='video', framerate=30, queueSize=30):
        """
        MediaPlayer initialization, sets the specified media type and
        any other related parameter.
        :param mediaType: the type of media content to be shown {image, video}
        :param framerate: for video types, the framerate at which it has to be played back
        :param queueSize: size of the video buffer, the larger the better (but uses more memory)
        """
        if mediaType not in ['image', 'video', 'webcam']:
            raise ValueError("MediaPlayer only supports 'image', 'video' and 'webcam' types")

        self.type = mediaType
        self.is_over = False
        self.is_stopped = False
        self.thread = None

        if mediaType in ['video', 'webcam']:
            # For playing back videos, a buffer and the framerate info are necessary
            self.rate = framerate
            self.queue = queue.Queue(maxsize=queueSize)


    def start(self):
        """
        Activate the media player, by starting a separate (daemon) thread that
        will take care of showing frames from the buffer, at the specified rate.
        This function does nothing for an 'image' type MediaPlayer
        """
        if self.type in ['video', 'webcam']:
            self.thread = Thread(target=self.update, args=(), daemon=False)
            self.thread.start()
        return self


    def update(self):
        """
        Thread main loop, that keeps grabbing frames from the video
        buffer (blocking operation if queue empty) and showing them
        on the screen at the desired framerate, until the user either
        pressed ESC or closes the player window with the mouse
        """

        if self.type == 'webcam':
            window_name = 'Webcam'
            pause = int(1000/self.rate)
        elif self.type == 'video':
            window_name = 'Video'
            # Playback videos slightly faster, to account for OS delays
            pause = int(1000/(1.1 * self.rate))

        # Keep looping until the video source runs out of content
        while not (self.is_over and self.queue.empty()):

            if self.is_stopped:
                break

            if not self.queue.empty():
                # Get the next frame from the video buffer
                frame = self.queue.get()

                # And show it on the screen
                cv.imshow(window_name, frame)

                # Wait for some milliseconds to pause the video stream
                # and also check if the user has pressed the ESC key to quit
                if cv.waitKey(pause) & 0xFF == ord('\x1b'):
                    self.stop()
                    sys.exit()

                # Check if the window has been closed by the user
                if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
                    self.stop()
                    sys.exit()

        self.stop()

        # Wait for one last action by the user before quitting
        while cv.getWindowProperty('Video', cv.WND_PROP_VISIBLE) > 0:
            if cv.waitKey(100) & 0xFF == ord('\x1b'):
                break


    def close(self):
        """
        Close the MediaPlayer window, waiting for it's termination
        """
        if not self.is_stopped:
            self.stop()

        # Wait for the termination of the MediaPlayer thread
        if self.type in ['video', 'webcam']:
            self.thread.join()
        elif self.type == 'image':
            # Wait for one last action by the user to close the window
            while cv.getWindowProperty('Image', cv.WND_PROP_VISIBLE) > 0:
                if cv.waitKey(100) & 0xFF == ord('\x1b'):
                    break

        # Close all GUI windows
        cv.destroyAllWindows()


    def show(self, frame):
        """
        Sends a single image or frame to the MediaPlayer, to have it shown on screen.
        If the type is set to 'image', the input is immediately shown on screen,
        otherwise, if the type is 'video' (recorded or live) the provided frame is
        pushed into the queue (blocking operation if full) and will be shown asap
        :param frame: the image or frame sent to the MediaPlayer
        """
        if self.is_over or self.is_stopped:
            return

        if self.type == 'image':
            # Show the image on the GUI
            cv.imshow('Image', frame)

        elif self.type in ['video', 'webcam']:
            # Enqueue the frame in the buffer, with 1s timeouts in order to not get
            # stuck if the video player is closed while the main thread is sleeping
            done = False
            while not done and not self.is_stopped:
                try:
                    self.queue.put(frame, timeout=1)
                    done = True
                except queue.Full:
                    done = False


    def stop(self):
        """
        Stops the execution of the MediaPlayer in its current state
        """
        self.is_stopped = True


    def signal_end(self):
        """
        Signals to the MediaPlayer that the multimedia content generated
        by the producer thread is finished, and therefore the player can
        terminate as soon as it empties the current queue.
        """
        self.is_over = True


    def stopped(self):
        """
        Returns true as soon as the MediaPlayer is stopped (by reaching
        the end of the content or by some user action)
        """
        return self.is_stopped or self.is_over


    def finished(self):
        """
        Returns true only if the player has reached the end of the content to be shown
        """
        return self.is_over and self.queue.empty()



class InputMedia:
    """
    The InputMedia class provides a convenient way of accessing different types
    of multimedia content (image, video or webcam stream) with a standard interface
    """

    def __init__(self, mediaType, path):
        """
        Initializes the InputMedia resource by setting some parameters and
        finally accessing the media source itself, checking for possible errors
        """
        if mediaType not in ['image', 'video', 'webcam']:
            raise ValueError("InputMedia type must be one of 'image', 'video' or 'webcam'")

        self.type = mediaType
        self.closed = False

        self.media = cv.imread(path) if mediaType == 'image' else cv.VideoCapture(path)
        if self.media is None:
            self.closed = True
            raise RuntimeError("Cannot open " + mediaType + " " + str(path))


    def close(self):
        """
        Releases the InputMedia resource, either a webcam or a video file.
        For image files this function has no effect, since no resource is kept open
        """
        if self.type in ['video', 'webcam'] and not self.closed:
            if self.media.isOpened:
                self.media.release()
            self.closed = True


    def read(self):
        """
        Reads data from the InputMedia source.
        In the case of video files or live webcam streams this returns the current frame in
        the sequence, while for 'image' types this keeps returning the source image each time
        In any case, the returned frame is resized to fit in 960x540 (preserving aspect ratio)
        """
        if self.closed:
            raise RuntimeError("Calling read() on a closed InputMedia instance")

        if self.type == 'image':
            frame = self.media
        elif self.type in ['video', 'webcam']:
            # Get the next frame from the video sequence or feed
            (grabbed, frame) = self.media.read()
            # Check if we have reached the end of the video file
            # or if the webcam has been disconnected
            if not grabbed:
                if self.media.isOpened:
                    self.media.release()
                self.closed = True
                return None

        # Get the image size
        height = frame.shape[0]
        width = frame.shape[1]

        # Resize the frame to 960x540 (preserve aspect ratio)
        factor = min(960 / width, 540 / height)
        new_size = (int(width * factor), int(height * factor))
        frame = cv.resize(frame, new_size)

        return frame


    def framerate(self):
        """
        Returns the framerate of the selected InputMedia source (0 for images)
        """
        if self.type == 'image':
            return 0
        if self.type == 'webcam':   # A fixed framerate of 30FPS is picked for webcams
            return 30
        return self.media.get(cv.CAP_PROP_FPS)


    def isOpened(self):
        """
        Returns true if the InputMedia has reached its end, whether it's by
        reaching EOF in the media file or by detecting that the webcam has been
        disconnected or turned off. Always returns true for 'image' types
        """
        return not self.closed
