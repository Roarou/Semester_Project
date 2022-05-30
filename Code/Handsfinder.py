import numpy as np
import mediapipe as mp
import cv2


def classifier(mhl, mhd, ldm, width=1920, height=1080):

    # Landmark initialization
    landmarks_x_R = []
    landmarks_y_R = []
    landmarks_z_R = []
    landmarks_x_L = []
    landmarks_y_L = []
    landmarks_z_L = []
    # First we check whether the first label is Right or Left, when it has been determined,
    # we check what if the latter has been detected twice,
    # if so we correct it , otherwise we check whether the second is left or right
    output_R = np.zeros((3, len(ldm)))
    output_L = np.zeros((3, len(ldm)))
    # Right Hand
    #print(mhd[0])
    if mhd[0].classification[0].label == 'Right':
        # Two indexes 1 or 0, we fill the matrix depending on the index
        idx = mhd[0].classification[0].index
        #1 is right 0 is left
        # Checking the list's length
        if (len(mhd) > 1 and mhd[0].classification[0].label
                == mhd[1].classification[0].label) or \
                (len(mhd) > 1 and mhd[0].classification[0].label
                    != mhd[1].classification[0].label):
            # If both hands are detected to be the same ,
            # Then take the first hand and fill the matrix with informations
            for i in ldm:
                landmarks_x_R.append(mhl[idx].landmark[i].x)
                landmarks_y_R.append(mhl[idx].landmark[i].y)
                landmarks_z_R.append(mhl[idx].landmark[i].z)

        else:
            for i in ldm:
                landmarks_x_R.append(mhl[0].landmark[i].x)
                landmarks_y_R.append(mhl[0].landmark[i].y)
                landmarks_z_R.append(mhl[0].landmark[i].z)

    if len(mhd) > 1 and mhd[1].classification[0].label \
            == 'Right' and mhd[0].classification[0].label \
                != mhd[1].classification[0].label:
        idx = mhd[1].classification[0].index
        for i in ldm:
            landmarks_x_R.append(mhl[idx].landmark[i].x)
            landmarks_y_R.append(mhl[idx].landmark[i].y)
            landmarks_z_R.append(mhl[idx].landmark[i].z)

    # Left hand
    if mhd[0].classification[0].label == 'Left':
        idx = mhd[0].classification[0].index
        if (len(mhd) > 1 and mhd[0].classification[0].label
                == mhd[1].classification[0].label) or \
                (len(mhd) > 1 and mhd[0].classification[0].label
                    != mhd[1].classification[0].label):
            for i in ldm:
                landmarks_x_L.append(mhl[idx].landmark[i].x)
                landmarks_y_L.append(mhl[idx].landmark[i].y)
                landmarks_z_L.append(mhl[idx].landmark[i].z)

        else:
            for i in ldm:
                landmarks_x_L.append(mhl[0].landmark[i].x)
                landmarks_y_L.append(mhl[0].landmark[i].y)
                landmarks_z_L.append(mhl[0].landmark[i].z)


    if len(mhd) > 1 and mhd[1].classification[0].label \
            == 'Left' and mhd[0].classification[0].label \
                != mhd[1].classification[0].label:

        idx = mhd[1].classification[0].index
        for i in ldm:
            landmarks_x_L.append(mhl[idx].landmark[i].x)
            landmarks_y_L.append(mhl[idx].landmark[i].y)
            landmarks_z_L.append(mhl[idx].landmark[i].z)

    # Must create a matrix instead with y axis being 0...21

    #Multiplying the lists by the width and height in order to get an image in the image frame
    landmarks_x_L = [value * width for value in landmarks_x_L]
    landmarks_y_L = [value * height for value in landmarks_y_L]
    landmarks_x_R = [value * width for value in landmarks_x_R]
    landmarks_y_R = [value * height for value in landmarks_y_R]
    if len(landmarks_z_R) != 0:
        output_R[0, :] = landmarks_x_R
        output_R[1, :] = landmarks_y_R
        output_R[2, :] = landmarks_z_R
    # print(self.results.multi_hand_landmarks[0].landmark[0].x)
    if len(landmarks_z_L) != 0:
        output_L[0, :] = landmarks_x_L
        output_L[1, :] = landmarks_y_L
        output_L[2, :] = landmarks_z_L

    return output_L, output_R