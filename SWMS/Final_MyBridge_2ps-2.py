

from pyduinobridge import Bridge_py
import pandas as pd
from datetime import datetime
import numpy as np
import statistics

serPort = '/dev/cu.usbmodem145101'
baudRate = 250000

myBridge = Bridge_py()
myBridge.begin(serPort, baudRate, numIntValues_FromPy=2, numFloatValues_FromPy=1)
myBridge.setSleepTime(2)

testData = []
testData.append("<BPM>")

raw_BPM_list = []

accept_range = [40-130]
ref_lever = 0
ref_BPM_counter = 0
ref_BPM1 = []
ref_BPM2 = []
ref_accept_list = []
ref_accept_var = 7

BPM_counter = 0
processed_BPM1 = []
processed_BPM2 = []
processed_accept_list = []
processed_BPM_list = []
accept_var = 0
var_flag = 0
old_BPM = 0
diff_flag = 0
accept_ref_diff = 5
not_enough_value_flag = 0

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


    raw_now = datetime.now()    # datetime object containing current date and time
    raw_dt_string = raw_now.strftime("%d/%m/%Y %H:%M:%S") # dd/mm/YY H:M:S
    raw_new_data = [BPM1, BPM2, raw_dt_string]
    raw_BPM_list.append(raw_new_data)
    raw_BPM_array = np.array(raw_BPM_list)
    raw_BPM_pd = pd.DataFrame(raw_BPM_array, columns =['BPM1', 'BPM2', 'Moment'])
    raw_BPM_pd.to_csv("RAW-BPMS.csv")

    ####################################################################################################################

    if ref_lever == 0:
        ref_BPM_counter += 1
        ref_BPM1.append(BPM1)
        ref_BPM2.append(BPM2)

        if ref_BPM_counter == 20:
            for i in range(len(ref_BPM1)):
                if ref_BPM1[i] > accept_range[0] and ref_BPM1[i] < accept_range[1]:
                    ref_accept_list.append(ref_BPM1[i])
            for i in range(len(ref_BPM2)):
                if ref_BPM2[i] > accept_range[0] and ref_BPM2[i] < accept_range[1]:
                    ref_accept_list.append(ref_BPM2[i])

            if len(ref_accept_list) >= 10:
                if np.var(ref_accept_list) <= ref_accept_var:
                    ref_accept_list = sorted(ref_accept_list)
                    del ref_accept_list[0]
                    del ref_accept_list[-1]
                    ref_BPM = np.average(ref_accept_list)
                    old_BPM = ref_BPM
                    ref_lever = 1
                else:
                  print("VAR REF FLAG")

            else:
                print("NOT ENOUGH VALUES FOR REF FLAG")
                ref_BPM_counter = 10

    ####################################################################################################################

    else:
        BPM_counter += 1
        processed_BPM1.append(BPM1)
        processed_BPM2.append(BPM2)

        if BPM_counter == 10:
            for i in range(len(processed_BPM1)):
                if processed_BPM1[i] > accept_range[0] and processed_BPM1[i] < accept_range[1]:
                    processed_accept_list.append(processed_BPM1[i])
            for i in range(len(processed_BPM2)):
                if processed_BPM2[i] > accept_range[0] and processed_BPM2[i] < accept_range[1]:
                    processed_accept_list.append(processed_BPM2[i])
            if len(processed_accept_list) >= 5:
                not_enough_value_flag = 0
                if np.var(processed_accept_list) >= accept_var:
                    var_flag = 0
                    potential_BPM = np.average(processed_accept_list)
                    if abs(potential_BPM-ref_BPM) <= accept_ref_diff:
                        diff_flag = 0
                        BPM = potential_BPM
                    else:
                        diff_flag += 1
                        BPM = old_BPM
                        if diff_flag >= 5:
                            print("DIFF FLAG")
                            ref_lever = 0
                else:
                    var_flag += 1
                    BPM = old_BPM
                    if var_flag >= 5:
                        print("VAR FLAG")
            else:
                not_enough_value_flag += 1
                BPM = old_BPM
                if not_enough_value_flag >= 5:
                        print("NOT ENOUGH VALUE FLAG")

            old_BPM = BPM
            BPM_counter = 0

        processed_now = datetime.now()    # datetime object containing current date and time
        processed_dt_string = processed_now.strftime("%d/%m/%Y %H:%M:%S") # dd/mm/YY H:M:S
        processed_new_data = [BPM, processed_dt_string]
        processed_BPM_list.append(processed_new_data)
        processed_BPM_array = np.array(processed_BPM_list)
        processed_BPM_pd = pd.DataFrame(processed_BPM_array, columns =['BPM', 'Moment'])
        processed_BPM_pd.to_csv("PROCESSED-BPMS.csv")

myBridge.close()
