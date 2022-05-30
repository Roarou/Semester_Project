import numpy as np
import pandas as pd
import os

def main(outR, outL, outrR, outlL, out_LR, out_LRXY, filename, currenttime=0,nb_fingers=2):

    #Concatenating both the Left and Right hands Data
    outLR = np.concatenate([(np.sqrt(outR[0, :] ** 2 + outR[1, :] ** 2)).reshape(1, nb_fingers),
                            (np.sqrt(outL[0, :] ** 2 + outL[1, :] ** 2)).reshape(1, nb_fingers)], axis=1).flatten()

    out_LR.append(outLR)
    outrR.append(np.sqrt(outR[0, :] ** 2 + outR[1, :] ** 2))
    outlL.append(np.sqrt(outL[0, :] ** 2 + outL[1, :] ** 2))

    #Get file with X and Y for outLR

    outLRXY = np.concatenate([(outR[0, :]).reshape(1, nb_fingers), outR[1, :].reshape(1, nb_fingers),
                              outL[0, :].reshape(1, nb_fingers), outL[1, :].reshape(1, nb_fingers)], axis=1).flatten()

    out_LRXY.append(outLRXY)
    oR = pd.DataFrame(data=outrR, columns=['Thumb_R', 'Index_R'])
    #['Wrist_R', 'Thumb_R', 'Index_R', 'Middle_R', 'Ring_R', 'Pinky_R'])
    oL = pd.DataFrame(data=outlL, columns=['Thumb_L', 'Index_L'])
    #['Wrist_L', 'Thumb_L', 'Index_L', 'Middle_L', 'Ring_L', 'Pinky_L'])
    oLR = pd.DataFrame(data=out_LR, columns=['Thumb_R', 'Index_R', 'Thumb_L', 'Index_L'])
    #Getting x and y coordinates instead of norm
    oLRXY = pd.DataFrame(data=out_LRXY, columns=['Thumb_RX', 'Index_RX', 'Thumb_RY', 'Index_RY', 'Thumb_LX', 'Index_LX', 'Thumb_LY', 'Index_LY'])
    #['Wrist_R', 'Thumb_R', 'Index_R', 'Middle_R', 'Ring_R', 'Pinky_R',
    #                                        'Wrist_L', 'Thumb_L', 'Index_L', 'Middle_L', 'Ring_L', 'Pinky_L'])
    #Adding a Time Columns
    oR['Time'] = currenttime
    oL['Time'] = currenttime
    oLR['Time'] = currenttime
    oLRXY['Time'] = currenttime
    path = os.getcwd()
    #Generating paths
    path_txt_out_L = path + '/OutputFileL/' + filename.replace('.mp4', '') + '_oL.txt'
    path_txt_out_R = path + '/OutputFileR/' + filename.replace('.mp4', '') + '_oR.txt'
    path_txt_out_LR = path + '/OutputFileLR/' + filename.replace('.mp4', '') + '_oLR.txt'
    path_txt_out_LRXY = path + '/OutputFileLRXY/' + filename.replace('.mp4', '') + '_oLRXY.txt'
    #Converting to csv file
    oL.to_csv(path_txt_out_L, index=False, sep='\t')
    oR.to_csv(path_txt_out_R, index=False, sep='\t')
    oLR.to_csv(path_txt_out_LR, index=False, sep='\t')
    oLRXY.to_csv(path_txt_out_LRXY, index=False, sep='\t')

