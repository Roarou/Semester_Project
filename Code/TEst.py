import os
import pandas as pd
import glob


def main():
    path = os.getcwd() + '/Exp1Normal04_distance.txt'
    df = pd.read_csv(path, sep='\t')
    path2 = os.getcwd() + '/oR.txt'
    df2 = pd.read_csv(path2, sep='\t')
    print(df.shape)
    print(df2.shape)
    fps = 25
    t1 = 4.2
    t2 = 11.84 - t1
    t3 = 14.92 - t2
    t4 =
if __name__ == "__main__":
    main()