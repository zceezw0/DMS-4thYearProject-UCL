import pandas as pd
from pandas import read_csv
import numpy as np
import random
import itertools
import os
import matplotlib.pyplot as plt

path = '/Users/rayan/PycharmProjects/4thYearProject/Final_BPM_Data/P003-P6-FINAL.csv'

data = read_csv(path)

list = []
print(data)
for i in range(len(data)):
    list.append(data.iloc[i][1])

del data['Unnamed: 0']
# Visualise with the year as x-absis
data.plot()
plt.title('SWMS BPM readings', fontsize=12)
plt.xlabel('Reading Frame', fontsize=10)
plt.ylabel('BPM', fontsize=10)
plt.ylim(60,120)
plt.grid()
plt.show()

