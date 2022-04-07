

import os
import pandas as pd
import random
import numpy as np

folder = '/Users/rayan/PycharmProjects/4thYearProject/Final_BS_BPM'
data = pd.read_excel('/Users/rayan/PycharmProjects/4thYearProject/Participation Information copy 2.xlsx')
x = 0
y = 0
l = []
for file in sorted(os.listdir(folder)):     #for all the files in dataset/image
        path = os.path.join(folder, file)
        for i in range(len(data)):
            if file[0:-4] == data.iloc[i][4]:
                print(file[0:-4],data.iloc[i][4])
                df1 = pd.read_csv(path)
                for j in range(len(df1)):
                    l.append(data.iloc[i][5])

                l_array = np.array(l)
                df2 = pd.DataFrame(l_array, columns =['label'])
                df1['label'] = df2['label']
                del df1['Unnamed: 0']
                try:
                    del df1['BPM POLAR']
                except KeyError:
                    print('no')
                df1.to_csv('/Users/rayan/PycharmProjects/4thYearProject/Final_DNN_Data/'+file[0:-4]+'.csv')

                l = []




