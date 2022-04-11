import os
import pandas as pd
directory = 'Data'
with open('test.txt','w') as outfile:
    for filename in os.listdir(directory):
        f = os.path.join(directory,filename)
        if os.path.isfile(f):
            with open(f) as infile:
                print(f)
                print(outfile)
                outfile.write(infile.read())
            outfile.write('\n')

dftrain1 = pd.read_csv(f,sep='\t')
print(dftrain1.head())
