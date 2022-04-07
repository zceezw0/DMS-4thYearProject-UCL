import os

if __name__ == '__main__':
    """
    The role of this script is to count the data in different labels,
    the distribution of its various features
    """
    data_root = r"D:\Chrome Download\CSCNN_dataset"
    features_root = os.path.join(data_root, "features")
    frames_root = os.path.join(data_root, "frames")

    # init label dict
    label_dicts = {}
    label_dict_value = {"yawn_count":[],
                        "blink_count":[],
                        "gaze_x":[],
                        "gaze_y":[],
                        "bpm_rayan":[],}
    for label in ["1","2","3"]:
        label_dicts[label] = label_dict_value

    # Traverse the data path and record the average value of each feature
    for root,dir,files in os.walk(features_root):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root,file)
                file_content = open(file_path).readlines()[1:]
                for line in file_content:
                    # load feature line by line
                    idx, time_second, blink_count, yawn_count, gaze_x, gaze_y, bpm_rayan, label = line.strip().split(",")
                    label_dicts[label]["yawn_count"].append(float(yawn_count))
                    label_dicts[label]["blink_count"].append(float(blink_count))
                    label_dicts[label]["gaze_x"].append(float(gaze_x))
                    label_dicts[label]["gaze_y"].append(float(gaze_y))
                    label_dicts[label]["bpm_rayan"].append(float(bpm_rayan))

    # show result
    for label in label_dicts.keys():
        for key in label_dicts[label].keys():
            print(key,sum(label_dicts[label][key])/len(label_dicts[label][key]))
        print("===========================")




