import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
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
        self.nb_landmarks = (0, 4, 8, 12, 16, 20)
#Drawing landmarks
    def handsFinder(self, image, draw=True,output_R = np.zeros((3, 6)),output_L = np.zeros((3, 6))):
        # Landmark initialization
        landmarks_x_R = []
        landmarks_y_R = []
        landmarks_z_R = []
        landmarks_x_L = []
        landmarks_y_L = []
        landmarks_z_L = []
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        mhl = self.results.multi_hand_landmarks
        mhd = self.results.multi_handedness
        ldm = self.nb_landmarks
        # Printing positions

        if mhl:
            self.j += 1
            # print(self.results.multi_handedness)

            # First we check whether the first label is Right or Left, when it has been determined,
            # we check what if the latter has been detected twice,
            # if so we correct it , otherwise we check whether the second is left or right
            # print(self.results.multi_handedness[0].classification[0].label,self.results.multi_handedness[0].classification[0].index)
            # idx = self.results.multi_handedness[0].classification[0].index
            # print(len(self.results.multi_hand_landmarks))

            # Right Hand
            if mhd[0].classification[0].label == 'Right':
                # Two indexes 1 or 0, we fill the matrix depending on the index
                idx = mhd[0].classification[0].index
                # Checking the list's length
                if len(mhd) > 1 and mhd[0].classification[0].label \
                        == mhd[1].classification[0].label:
                    # If both hands are detected to be the same ,
                    # Then take the first hand and fill the matrix with informations
                    for i in ldm:
                        landmarks_x_R.append(mhl[idx].landmark[i].x)
                        landmarks_y_R.append(mhl[idx].landmark[i].y)
                        landmarks_z_R.append(mhl[idx].landmark[i].z)
                        break
                else:
                    for i in ldm:
                        landmarks_x_R.append(mhl[0].landmark[i].x)
                        landmarks_y_R.append(mhl[0].landmark[i].y)
                        landmarks_z_R.append(mhl[0].landmark[i].z)
                        break

            elif len(mhd) > 1 and mhd[1].classification[0].label \
                    == 'Right':
                idx = mhd[1].classification[0].index
                for i in ldm:
                    landmarks_x_R.append(mhl[idx].landmark[i].x)
                    landmarks_y_R.append(mhl[idx].landmark[i].y)
                    landmarks_z_R.append(mhl[idx].landmark[i].z)

            # Left hand
            if mhd[0].classification[0].label == 'Left':
                idx = mhd[0].classification[0].index
                if len(mhd) > 1 and mhd[0].classification[0].label \
                        == mhd[1].classification[0].label:
                    for i in ldm:
                        landmarks_x_L.append(mhl[idx].landmark[i].x)
                        landmarks_y_L.append(mhl[idx].landmark[i].y)
                        landmarks_z_L.append(mhl[idx].landmark[i].z)
                        break
                else:
                    for i in ldm:
                        landmarks_x_L.append(mhl[0].landmark[i].x)
                        landmarks_y_L.append(mhl[0].landmark[i].y)
                        landmarks_z_L.append(mhl[0].landmark[i].z)
                        break

            elif len(mhd) > 1 and mhd[1].classification[0].label \
                    == 'Left':
                idx = mhd[1].classification[0].index
                for i in ldm:
                    landmarks_x_L.append(mhl[idx].landmark[i].x)
                    landmarks_y_L.append(mhl[idx].landmark[i].y)
                    landmarks_z_L.append(mhl[idx].landmark[i].z)

            # Must create a matrix instead with y axis being 0...21
            output_R = np.zeros((3, len(ldm)))
            output_L = np.zeros((3, len(ldm)))
            if len(landmarks_z_R) != 0:
                output_R[0, :] = landmarks_x_R
                output_R[1, :] = landmarks_y_R
                output_R[2, :] = landmarks_z_R
            # print(self.results.multi_hand_landmarks[0].landmark[0].x)
            if len(landmarks_z_L) != 0:
                output_L[0, :] = landmarks_x_L
                output_L[1, :] = landmarks_y_L
                output_L[2, :] = landmarks_z_L
            # print(output)

            for handLms in mhl:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image, output_R*image.shape.w, output_L*image.shape.w

    #Finding hands and positions

    def positionFinder(self,image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # Creates a list with all the landmarks in the image frame
                lmlist.append([id, cx, cy])
            if draw:
                #Origin top left
                cv2.circle(image,   (cx, cy), 15 , (255, 0, 255), cv2.FILLED)

        return lmlist, self.j


def main():
    #Using Video
    cap = cv2.VideoCapture('Exp1Normal01.mp4')
    #Executing HandTracking Script
    tracker = handTracker()
    #Saving Video

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))
    
    outrR = []
    outlL = []

    while True:
        success, image = cap.read()
        image,  outR, outL = tracker.handsFinder(image)
        lmList,j = tracker.positionFinder(image)
        if len(lmList) != 0:
            #Prints the position and ID of a certain landmark , in this case the thumb tip
            print(lmList[4])
        outrR.append(outR[0,:])
        outlL.append(outL[0,:])
        writer.write(image)
        oL = pd.DataFrame(data=outlL, columns=['Wrist', 'Thumb', 'Index', 'Middle', 'Ring', 'Pinky'])
        oR = pd.DataFrame(data=outrR, columns=['Wrist', 'Thumb', 'Index', 'Middle', 'Ring', 'Pinky'])

        oL.to_csv('ol.txt', index=False)
        oR.to_csv('oR.txt', index=False)
        cv2.imshow("Video", image)
        cv2.waitKey(1)
        #print(j)

    #print(int(j/61))
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()