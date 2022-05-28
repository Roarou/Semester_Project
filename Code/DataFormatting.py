import os
import pandas as pd
from Closest_neighbour import nearest_neighbor
import numpy as np
#In this script we are trying to format the data. Indeed, we change our HandTrackingModule output of length n to the
# labeled data size x by finding a mean value


def data_formatting(filename_actions, filename_toformat):
    path = os.getcwd() + filename_actions
    df = pd.read_csv(path, sep='\t')
    path2 = os.getcwd() + filename_toformat
    df2 = pd.read_csv(path2, sep='\t')
    Fingers = df2.columns.tolist()
    #Removing Time from Fingers list
    Fingers.pop(-1)
    df['Action'] = df['Action'].astype('str')

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
        q_5 = df2[finger].quantile(0.50)
        q_10 = df2[finger].quantile(0.10)
        df2[finger] = np.where(df2[finger] < q_10, q_5, df2[finger])

    df2['Action'] = df['Action']

    df2.to_csv(path2, index=False, sep='\t')


if __name__ == "__main__":
    Fingers = ['Thumb_L',	'Index_L']
    data_formatting(Fingers, '/Exp1Normal01_distance.txt', '/Exp1Normal01.mp4_oL.txt')