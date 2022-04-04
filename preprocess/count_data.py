import json
import os
import pandas as pd

if __name__ == '__main__':
    data_root = r"D:\Chrome Download\CSCNN_dataset"
    features_root = os.path.join(data_root, "features")
    frames_root = os.path.join(data_root, "frames")

    label1_dict = {"yawn_count":[],
                   "blink_count":[],
                   "gaze_x":[],
                   "gaze_y":[],
                   "bpm_rayan":[],}
    label2_dict = {"yawn_count": [],
                   "blink_count": [],
                   "gaze_x": [],
                   "gaze_y": [],
                   "bpm_rayan": [], }
    label3_dict = {"yawn_count": [],
                   "blink_count": [],
                   "gaze_x": [],
                   "gaze_y": [],
                   "bpm_rayan": [], }

    for root,dir,files in os.walk(features_root):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root,file)
                file_content = open(file_path).readlines()[1:]
                for line in file_content:
                    idx, time_second, blink_count, yawn_count, gaze_x, gaze_y, bpm_rayan, label = line.strip().split(",")
                    if label == "1":
                        label1_dict["yawn_count"].append(float(yawn_count))
                        label1_dict["blink_count"].append(float(blink_count))
                        label1_dict["gaze_x"].append(float(gaze_x))
                        label1_dict["gaze_y"].append(float(gaze_y))
                        label1_dict["bpm_rayan"].append(float(bpm_rayan))
                    if label == "2":
                        label2_dict["yawn_count"].append(float(yawn_count))
                        label2_dict["blink_count"].append(float(blink_count))
                        label2_dict["gaze_x"].append(float(gaze_x))
                        label2_dict["gaze_y"].append(float(gaze_y))
                        label2_dict["bpm_rayan"].append(float(bpm_rayan))
                    if label == "3":
                        label3_dict["yawn_count"].append(float(yawn_count))
                        label3_dict["blink_count"].append(float(blink_count))
                        label3_dict["gaze_x"].append(float(gaze_x))
                        label3_dict["gaze_y"].append(float(gaze_y))
                        label3_dict["bpm_rayan"].append(float(bpm_rayan))
    dicts = [label1_dict,label2_dict,label3_dict]
    for label_dict in dicts:
        for key in label_dict.keys():
            print(key,sum(label_dict[key])/len(label_dict[key]))
        print("=========")
    pass




