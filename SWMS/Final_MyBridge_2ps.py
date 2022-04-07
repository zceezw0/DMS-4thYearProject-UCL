

serPort = '/dev/cu.usbmodem1434301'
baudRate = 250000

from pyduinobridge import Bridge_py
import pandas as pd
from datetime import datetime
import numpy as np

myBridge = Bridge_py()
myBridge.begin(serPort, baudRate, numIntValues_FromPy=2, numFloatValues_FromPy=1)
myBridge.setSleepTime(2)

testData = []
testData.append("<BPM>")

BPM_list = []

while True:
    dataFromArduino = myBridge.writeAndRead_Strings(testData)
    #myBridge.setVerbosity(0)
    dataFromArduino = dataFromArduino[0]    #disregard the encapsulating list

    for i in range(len(dataFromArduino)):
        if dataFromArduino[i:i+2]=='#0':
            BPM_inter = dataFromArduino[i+3:]
            for j in range(len(BPM_inter)):
                if BPM_inter[j:j+2]=='In':
                    BPM1 = BPM_inter[:j-1]

    for i in range(len(dataFromArduino)):
        if dataFromArduino[i:i+2]=='#1':
            BPM_inter = dataFromArduino[i+3:]
            for j in range(len(BPM_inter)):
                if BPM_inter[j:j+2]=='Ti':
                    BPM2 = BPM_inter[:j-1]


    now = datetime.now()    # datetime object containing current date and time
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S") # dd/mm/YY H:M:S

    new_data = [BPM1, BPM2, dt_string]
    BPM_list.append(new_data)
    BPM_array = np.array(BPM_list)
    BPM_pd = pd.DataFrame(BPM_array, columns =['BPM1', 'BPM2', 'Moment'])

    BPM_pd.to_csv("X.csv")

myBridge.close()

