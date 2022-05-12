import numpy as np
import mediapipe as mp
import cv2


def classifier(mhl, mhd, ldm):

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

    return landmarks_x_L, landmarks_y_L, landmarks_z_L, landmarks_x_R, landmarks_y_R, landmarks_z_R