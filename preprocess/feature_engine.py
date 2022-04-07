import os
import json

from tqdm import tqdm

if __name__ == '__main__':
    """
    The role of this script is to obtain discrete categorical features by 
    bucketing the original continuous numerical features
    """
    # root of each data
    data_root = r"D:\Chrome Download\CSCNN_dataset"
    features_root = os.path.join(data_root, "features")
    frames_root = os.path.join(data_root, "processed_frames")

    # init variables
    test_names = ["P027-P1", "P028-P1", "P029-P1", "P030-P1", "P031-P1"]
    train = []
    test = []
    blinks = []
    yawns = []
    gaze_xs = []
    gaze_ys = []
    hearts = []

    # Traverse the data path
    for root, dir, files in os.walk(features_root):
        for file in files:
            if file.endswith(".csv"):
                avi_name = file.split(".")[0]
                file_path = os.path.join(root, file)
                file_content = open(file_path).readlines()[1:]
                for line in file_content:
                    json_content = {}
                    # load feature line by line
                    idx, time_second, blink_count, yawn_count, gaze_x, gaze_y, bpm_rayan, label = line.strip().split(",")

                    """
                    The criterion for our data bucketing is to make 
                    continuous features evenly fall into each bucket.
                    """

                    """
                    blink is divided into 5 categories, 
                    the bounds of each bucket are [1, 3, 5, 8]
                    """
                    # category : 5
                    blink_count = int(blink_count)
                    blink_feature = 0
                    if blink_count <= 1:
                        blink_feature = 0
                    if blink_count > 1 and blink_count <= 3:
                        blink_feature = 1
                    if blink_count > 3 and blink_count <= 5:
                        blink_feature = 2
                    if blink_count > 5 and blink_count <= 8:
                        blink_feature = 3
                    if blink_count > 8:
                        blink_feature = 4

                    """
                    blink is divided into 2 categories, 
                    the bounds of each bucket are [0]
                    """
                    # category : 2
                    yawn_count = int(yawn_count)
                    yawn_feature = 0
                    if yawn_count == 0:
                        yawn_feature = 0
                    if yawn_count > 0:
                        yawn_feature = 1

                    """
                    blink is divided into 5 categories, 
                    the bounds of each bucket are [4.24, 5.51, 6.95, 8.84]
                    """
                    # category : 5
                    gaze_x = float(gaze_x)
                    gazex_feature = 0
                    if gaze_x < 4.24:
                        gazex_feature = 0
                    if gaze_x >= 4.24 and gaze_x < 5.51:
                        gazex_feature = 1
                    if gaze_x >= 5.51 and gaze_x < 6.95:
                        gazex_feature = 2
                    if gaze_x >= 6.95 and gaze_x < 8.84:
                        gazex_feature = 3
                    if gaze_x >= 8.84:
                        gazex_feature = 4

                    """
                    blink is divided into 5 categories, 
                    the bounds of each bucket are [2.66, 3.75, 4.83, 6.48]
                    """
                    # category : 5
                    gaze_y = float(gaze_y)
                    gazey_feature = 0
                    if gaze_y < 2.66:
                        gazey_feature = 0
                    if gaze_y >= 2.66 and gaze_y < 3.75:
                        gazey_feature = 1
                    if gaze_y >= 3.75 and gaze_y < 4.83:
                        gazey_feature = 2
                    if gaze_y >= 4.83 and gaze_y < 6.48:
                        gazey_feature = 3
                    if gaze_y >= 6.48:
                        gazey_feature = 4

                    """
                    blink is divided into 5 categories, 
                    the bounds of each bucket are [68, 72, 76, 80]
                    """
                    # category : 5
                    heart = int(bpm_rayan)
                    heart_feature = 0
                    if heart < 68:
                        heart_feature = 0
                    if heart >= 68 and heart < 72:
                        heart_feature = 1
                    if heart >= 72 and heart < 76:
                        heart_feature = 2
                    if heart >= 76 and heart < 80:
                        heart_feature = 3
                    if heart >= 80:
                        heart_feature = 4

                    json_content["blink"] = blink_feature
                    json_content["yawn"] = yawn_feature
                    json_content["gaze_x"] = gazex_feature
                    json_content["gaze_y"] = gazey_feature
                    json_content["heart"] = heart_feature
                    json_content["label"] = int(label)
                    json_content["image_path"] = os.path.join(avi_name, "{}_{}.jpg".format(avi_name, idx))
                    json_content = json.dumps(json_content)

                    blinks.append(int(blink_count))
                    yawns.append(int(yawn_count))
                    gaze_xs.append(float(gaze_x))
                    gaze_ys.append(float(gaze_y))
                    hearts.append(int(bpm_rayan))

                    if avi_name in test_names:
                        test.append(json_content)
                    else:
                        train.append(json_content)
    blinks.sort()
    yawns.sort()
    gaze_xs.sort()
    gaze_ys.sort()
    hearts.sort()

    # average bucket
    print("blinks", blinks[1300], blinks[2600], blinks[3900], blinks[5200])
    print("yawns", yawns[1300], yawns[2600], yawns[3900], yawns[5200])
    print("gaze_xs", gaze_xs[1300], gaze_xs[2600], gaze_xs[3900], gaze_xs[5200])
    print("gaze_ys", gaze_ys[1300], gaze_ys[2600], gaze_ys[3900], gaze_ys[5200])
    print("hearts", hearts[1300], hearts[2600], hearts[3900], hearts[5200])

    # write to new train json
    with open(r"D:\Chrome Download\CSCNN_dataset\train.json", "w") as F:
        for line in tqdm(train):
            F.writelines("{}\n".format(line))
    F.close()

    # write to new test json
    with open(r"D:\Chrome Download\CSCNN_dataset\test.json", "w") as F:
        for line in tqdm(test):
            F.writelines("{}\n".format(line))
    F.close()