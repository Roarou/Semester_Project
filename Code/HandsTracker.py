import cv2
import numpy as np
from Handsfinder import classifier
import mediapipe as mp

class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.j = 0
        self.nb_landmarks = (4, 8) #(0, 4, 8, 12, 16, 20)
#Drawing landmarks
    def handsFinder(self, image, draw=True, output_R = np.zeros((3, 2)),output_L = np.zeros((3, 2))):
        # Landmark initialization
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Getting the current hands' data
        self.results = self.hands.process(imageRGB)
        #multi_hand_landmarks field that contains the hand landmarks on each detected hand.
        mhl = self.results.multi_hand_landmarks
        #multi_handedness field that contains the handedness (left v.s.right hand) of the detected hand.
        mhd = self.results.multi_handedness
        #What landmarks you want to use
        ldm = self.nb_landmarks
        height, width, c = image.shape
        # Printing positions

        if mhl:
            self.j += 1
            # print(self.results.multi_handedness)

            # First we check whether the first label is Right or Left, when it has been determined,
            # we check what if the latter has been detected twice,
            # if so we correct it , otherwise we check whether the second is left or right
            # Right Hand
            #Outputs array with
            output_R, output_L =classifier(mhl, mhd, ldm, width, height)
            #Approach 1: Vector Norm sqrt(x^2+y^2)
            norm_R = np.sqrt(output_R[0, :]**2 + output_R[1, :]**2)
            norm_L = np.sqrt(output_L[0, :]**2 + output_L[1, :]**2)
            #print(norm_R.shape)
            #print("test")
            #Drawing hands
            for handLms in mhl:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image, output_R, output_L