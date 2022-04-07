

import os
import pandas as pd
import random

folder = '/Users/rayan/PycharmProjects/4thYearProject/Final_BS'
folder2 = '/Users/rayan/PycharmProjects/4thYearProject/Final_BPM_Data'

for file in sorted(os.listdir(folder)):     #for all the files in dataset/image
        if file == '.DS_Store':
            continue
        path = os.path.join(folder, file)
        for file2 in sorted(os.listdir(folder2)):     #for all the files in dataset/image
            if file2 == '.DS_Store':
                continue
            pat2 = os.path.join(folder2, file2)
            if file[0:-10] == file2[0:-10]:
                print(file[0:-10])
                df1 = pd.read_excel(path)
                #print(df1)
                df2 = pd.read_csv(pat2)
                mean = int(df2['BPM'].mean())

                while len(df1) > len(df2):
                    df3 = pd.DataFrame({"Unnamed: 0":[-1],"BPM":[random.randrange(mean-5, mean+5)], "Moment":[-1]})
                    df2 = df2.append(df3, ignore_index=True)

                #print(df2['BPM'])
                df1['BPM RAYAN'] = df2['BPM']
                #print(df1)
                df1.to_csv('/Users/rayan/PycharmProjects/4thYearProject/Final_BS_BPM/'+file[0:-10]+'.csv')



