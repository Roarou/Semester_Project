import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Exp1PrimingApplyTip.mp4_oLR.txt', sep='\t')

print(df['Thumb_R'].skew())

print(df['Thumb_R'].describe())
print(df['Thumb_R'].quantile(0.22))
print(df['Thumb_R'].quantile(0.90))
q_5 = df['Thumb_R'].quantile(0.50)
q_95 = df['Thumb_R'].quantile(0.10)
df['Thumb_R'] = np.where(df['Thumb_R'] < q_95, q_5, df['Thumb_R'])
print(df['Thumb_R'].skew())
print(df['Thumb_R'].describe())

df.to_csv('Exp1PrimingApplyTip.mp4_oL.txt', index=False, sep='\t')
Fingers = df.columns.tolist()
#Removing Time from Fingers list
Fingers.pop(-1)
for i, finger in enumerate(Fingers):

    # Replacing Outliers with Median Values AKA 0 values due to lack of detection
    q_5 = df[finger].quantile(0.50)
    q_10 = df[finger].quantile(0.10)
    df[finger] = np.where(df[finger] < q_10, q_5, df[finger])
    print(df[finger])