import numpy as np
import pandas as pd
import os

def main(outR, outL, outrR, outlL, out_LR, filename, currenttime=0,nb_fingers=2):

    #Concatenating both the Left and Right hands Data
    outLR = np.concatenate([(np.sqrt(outR[0, :] ** 2 + outR[1, :] ** 2)).reshape(1, nb_fingers),
                            (np.sqrt(outL[0, :] ** 2 + outL[1, :] ** 2)).reshape(1, nb_fingers)], axis=1).flatten()

    out_LR.append(outLR)
    outrR.append(np.sqrt(outR[0, :] ** 2 + outR[1, :] ** 2))
    outlL.append(np.sqrt(outL[0, :] ** 2 + outL[1, :] ** 2))

    oR = pd.DataFrame(data=outrR, columns=['Thumb_R', 'Index_R'])
    #['Wrist_R', 'Thumb_R', 'Index_R', 'Middle_R', 'Ring_R', 'Pinky_R'])
    oL = pd.DataFrame(data=outlL, columns=['Thumb_L', 'Index_L'])
    #['Wrist_L', 'Thumb_L', 'Index_L', 'Middle_L', 'Ring_L', 'Pinky_L'])
    oLR = pd.DataFrame(data=out_LR, columns=['Thumb_R', 'Index_R', 'Thumb_L', 'Index_L'])
    #['Wrist_R', 'Thumb_R', 'Index_R', 'Middle_R', 'Ring_R', 'Pinky_R',
    #                                        'Wrist_L', 'Thumb_L', 'Index_L', 'Middle_L', 'Ring_L', 'Pinky_L'])
    #Adding a Time Columns
    oR['Time'] = currenttime
    oL['Time'] = currenttime
    oLR['Time'] = currenttime

    path = os.getcwd()
    #Generating paths
    path_txt_out_L = path + '/OutputFileL/' + filename + '_oL.txt'
    path_txt_out_R = path + '/OutputFileR/' + filename + '_oR.txt'
    path_txt_out_LR = path + '/OutputFileLR/' + filename + '_oLR.txt'
    #Converting to csv file
    oL.to_csv(path_txt_out_L, index=False, sep='\t')
    oR.to_csv(path_txt_out_R, index=False, sep='\t')
    oLR.to_csv(path_txt_out_LR, index=False, sep='\t')
