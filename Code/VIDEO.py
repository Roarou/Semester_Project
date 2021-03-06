#import needed packages
import cv2
import math

#This class provides a way to easily access the basic parameters of the video
#Also it adds a utility function to display progress bar on the video when
#displayed with OpenCV imshow function


class utilsVideo:
    #Ctor that takes following arguments
    #capture - video capture class object from OpenCV
    def __init__(self, capture):
        self.capture = capture

    def getStats(self):
        # get frames per seconds (fps) from the video
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        # total frame counts
        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # total duraiton of video in milli seconds
        durationSec = frame_count / fps
        return (fps, frame_count, durationSec)

    def displayProgressBar(self, currentframe):
        # get next frame number out of all the frames for video
        nextFrameNo = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
        # get total number of frames in the video
        totalFrames = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        # calculate the percent complete based on the frame currently
        # playing. OpenCV does provide a variable to access this
        # property directly (CAP_PROP_POS_AVI_RATIO), however
        # it seems to not work all the time, hence we calculate internally
        complete = nextFrameNo / totalFrames

        # progress bar thickness
        lineThickness = 25
        # progress bar will be displayed 4% from the bottom of the frame
        y = math.ceil(currentframe.shape[1] - currentframe.shape[1] / 4)
        # display progress bar across the width of the video
        x = 0
        w = currentframe.shape[0]
        # white line as background for progressbar
        cv2.line(currentframe, (x, y), (w, y), (255, 255, 255), lineThickness)
        # red line as progress on top of that
        cv2.line(currentframe, (x, y), (math.ceil(w * complete), y), (0, 0, 255), lineThickness)
        return currentframe