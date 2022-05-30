import os
import pandas as pd
from Closest_neighbour import nearest_neighbor
import numpy as np
#In this script we are trying to format the data. Indeed, we change our HandTrackingModule output of length n to the
# labeled data size x by finding a mean value


def data_formatting(filename_actions, filename_toformat, filename_combine, choice='All'):
    path = os.getcwd() + filename_actions
    df = pd.read_csv(path, sep='\t')
    path2 = os.getcwd() + filename_toformat
    df2 = pd.read_csv(path2, sep='\t')
    pathcomb = os.getcwd() + filename_combine
    dfcomb = pd.read_csv(pathcomb, sep='\t')
    Fingers = df2.columns.tolist()
    #Removing Time from Fingers list
    Fingers.pop(-1)
    df['Action'] = df['Action'].astype('str')
    df = df.drop('sequence', axis=1)
    df = df.drop('Dict_value', axis=1)
    #print(df)
    finger_storage = []
    for i, finger in enumerate(Fingers):
        finger_n = []

        for i in range(df.shape[0]):
            #Here we find the nearest neighbour to our current time
            nearest_start, id_start = nearest_neighbor(df.start_time[i], df2.Time, 1)
            nearest_end, id_end = nearest_neighbor(df.end_time[i], df2.Time, 1)
            #We convert the ids to int
            id_start = int(id_start[0])
            id_end = int(id_end[0])
            #Going through the array
            dft = df2[finger].to_numpy()
            dft = dft[id_start:id_end].mean()
            finger_n.append(dft)

        finger_storage.append(finger_n)
   #print(finger_storage)
    df2 = pd.DataFrame()
    for i, finger in enumerate(Fingers):
        #print(len(finger_storage[i]))
        df2[finger] = finger_storage[i]
        #Replacing Outliers with Median Values AKA 0 values due to lack of detection
        q1 = np.percentile(df2[finger], 25)
        q3 = np.percentile(df2[finger], 75)
        # print(q1, q3)
        IQR = q3 - q1
        lwr_bound = q1 - (1.5 * IQR)
        #upr_bound = q3 + (1.5 * IQR)
        #First approach
        q_5 = df2[finger].quantile(0.50)
        #q_10 = df2[finger].quantile(0.10)
        df2[finger] = np.where(df2[finger] < lwr_bound, q_5, df2[finger])


    for col in df.columns:
        df2[col] = df[col]
        df = df.drop(col, axis=1)



    if choice == 'All':
        df2['Thumb_RX'] = df2['Thumb_RX'] - dfcomb['Visual Intake Position X [px]']
        df2['Index_RX'] = df2['Index_RX'] - dfcomb['Visual Intake Position X [px]']
        df2['Thumb_LX'] = df2['Thumb_LX'] - dfcomb['Visual Intake Position X [px]']
        df2['Index_LX'] = df2['Index_LX'] - dfcomb['Visual Intake Position X [px]']
        df2['Thumb_RY'] = df2['Thumb_RY'] - dfcomb['Visual Intake Position X [px]']
        df2['Index_RY'] = df2['Index_RY'] - dfcomb['Visual Intake Position X [px]']
        df2['Thumb_LY'] = df2['Thumb_LY'] - dfcomb['Visual Intake Position X [px]']
        df2['Index_LY'] = df2['Index_LY'] - dfcomb['Visual Intake Position X [px]']
        df2['Thumb_R'] = (df2['Thumb_RX'] ** 2 + df2['Thumb_RY'] ** 2) ** (1 / 2)
        df2['Thumb_L'] = (df2['Thumb_LX'] ** 2 + df2['Thumb_LY'] ** 2) ** (1 / 2)
        df2['Index_R'] = (df2['Index_RX'] ** 2 + df2['Index_RY'] ** 2) ** (1 / 2)
        df2['Index_L'] = (df2['Index_LX'] ** 2 + df2['Index_LY'] ** 2) ** (1 / 2)
        df2 = df2.drop(['Thumb_RX', 'Index_RX', 'Thumb_RY', 'Index_RY', 'Thumb_LX', 'Index_LX', 'Thumb_LY', 'Index_LY'], axis=1)
    if choice == 'Right':
        df2['Thumb_RX'] = df2['Thumb_RX'] - dfcomb['Visual Intake Position X [px]']
        df2['Index_RX'] = df2['Index_RX'] - dfcomb['Visual Intake Position X [px]']
        df2['Thumb_RY'] = df2['Thumb_RY'] - dfcomb['Visual Intake Position X [px]']
        df2['Index_RY'] = df2['Index_RY'] - dfcomb['Visual Intake Position X [px]']
        df2['Thumb_R'] = (df2['Thumb_RX'] ** 2 + df2['Thumb_RY'] ** 2) ** (1 / 2)
        df2['Index_R'] = (df2['Index_RX'] ** 2 + df2['Index_RY'] ** 2) ** (1 / 2)
        df2 = df2.drop(['Thumb_RX', 'Index_RX', 'Thumb_RY', 'Index_RY'], axis=1)
    if choice == 'Left':
        df2['Thumb_LX'] = df2['Thumb_LX'] - dfcomb['Visual Intake Position X [px]']
        df2['Index_LX'] = df2['Index_LX'] - dfcomb['Visual Intake Position X [px]']
        df2['Thumb_LY'] = df2['Thumb_LY'] - dfcomb['Visual Intake Position X [px]']
        df2['Index_LY'] = df2['Index_LY'] - dfcomb['Visual Intake Position X [px]']
        df2['Thumb_L'] = (df2['Thumb_LX'] ** 2 + df2['Thumb_LY'] ** 2) ** (1 / 2)
        df2['Index_L'] = (df2['Index_LX'] ** 2 + df2['Index_LY'] ** 2) ** (1 / 2)
        df2 = df2.drop(['Thumb_LX', 'Index_LX', 'Thumb_LY', 'Index_LY'], axis=1)

    df2.to_csv(path2, index=False, sep='\t')
if __name__ == "__main__":
    Fingers = ['Thumb_L',	'Index_L', 'Thumb_R',	'Index_R']
    data_formatting('/Data/Exp1Normal02_distance.txt', '/OutputFileLR/Exp1Normal02_oLR.txt')