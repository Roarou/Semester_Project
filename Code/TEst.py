import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Exp1PrimingApplyTip.mp4_oL.txt', sep='\t')

print(df['Thumb_L'].skew())

print(df['Thumb_L'].describe())
print(df['Thumb_L'].quantile(0.10))
print(df['Thumb_L'].quantile(0.90))
q_5 = df['Thumb_L'].quantile(0.50)
q_95 = df['Thumb_L'].quantile(0.10)
df['Thumb_L'] = np.where(df['Thumb_L'] < q_95, q_5, df['Thumb_L'])
print(df['Thumb_L'].skew())
print(df['Thumb_L'].describe())

df.to_csv('Exp1PrimingApplyTip.mp4_oL.txt', index=False, sep='\t')