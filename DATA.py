import numpy as np
import pandas as pd


def main(outR, outL, outrR, outlL, out_LR, currenttime=0):

    outLR = np.concatenate([(np.sqrt(outR[0, :] ** 2 + outR[1, :] ** 2)).reshape(1, 6),
                            (np.sqrt(outL[0, :] ** 2 + outL[1, :] ** 2)).reshape(1, 6)], axis=1).flatten()

    out_LR.append(outLR)
    outrR.append(np.sqrt(outR[0, :] ** 2 + outR[1, :] ** 2))
    outlL.append(np.sqrt(outL[0, :] ** 2 + outL[1, :] ** 2))

    oR = pd.DataFrame(data=outrR, columns=['Wrist_R', 'Thumb_R', 'Index_R', 'Middle_R', 'Ring_R', 'Pinky_R'])
    oL = pd.DataFrame(data=outlL, columns=['Wrist_L', 'Thumb_L', 'Index_L', 'Middle_L', 'Ring_L', 'Pinky_L'])
    oLR = pd.DataFrame(data=out_LR, columns=['Wrist_R', 'Thumb_R', 'Index_R', 'Middle_R', 'Ring_R', 'Pinky_R',
                                             'Wrist_L', 'Thumb_L', 'Index_L', 'Middle_L', 'Ring_L', 'Pinky_L'])
    oR['Time'] = currenttime
    oL['Time'] = currenttime
    oLR['Time'] = currenttime

    oL.to_csv('oL.txt', index=False, sep='\t')
    oR.to_csv('oR.txt', index=False, sep='\t')
    oLR.to_csv('oLR.txt', index=False, sep='\t')

