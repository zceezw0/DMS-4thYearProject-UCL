import sys
import os
from glob import glob
from PySide2 import QtWidgets,QtCore,QtGui
from PySide2.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PySide2.QtCore import QDir, QTimer,Slot
from PySide2.QtGui import QPixmap,QImage
from ui_mainwindow import Ui_MainWindow
import cv2
from face_processor import FaceProcessor
import argparse
import os
import copy
import csv
import codecs
from pathlib import Path
import numpy as np
from tqdm import tqdm

global count

count = 0

EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 2

MAR_THRESH = 0.65
MOUTH_AR_CONSEC_FRAMES = 3

COUNTER = 0
TOTAL = 0
hTOTAL = 0
mCOUNTER = 0
hmTOTAL = 0
mTOTAL = 0
ActionCOUNTER = 0

Roll = 0

Gaze_Vectors_X = []
Gaze_Vectors_Y = []

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.f_type = 0

    def window_init(self):
        # self.label.setText("Please turn on the camera!")
        # self.label_2.setText("Fatigue Detect:")
        self.label_3.setText("Blink Count:0")
        self.label_4.setText("Yawn Count:0")
        self.label_5.setText("")
        self.label_9.setText("Yawn Rate:0")
        self.label_10.setText("Blink Rate:0")
        self.label_13.setText("Gaze X: 0")
        self.label_14.setText("Gaze Y: 0")
        self.menu.setTitle("Turn on Camera")
        self.actionOpen_camera.setText("Turn on Camera")
        self.actionOpen_camera.triggered.connect(CamConfig_init)
        self.label.setScaledContents(True)

class CamConfig:
    def __init__(self):
        self.face_processor = FaceProcessor()
        self.eye_threshold = 0
        self.mouth_threshold = 0

        Ui_MainWindow.printf(window,"Turning on the camera. Please wait...")

        self.v_timer = QTimer()

        self.time_number = 0
        self.cap = cv2.VideoCapture(0)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.time_stamp = time_step * self.fps

        if not self.cap:
            Ui_MainWindow.printf(window,"Failed to turn on camera.")
            return

        print("[INFO] Please keep your eyes open and your mouth closed for a while!")
        for _ in range(50):
            self.init_threshold()

        self.eye_threshold /= 50
        self.mouth_threshold /= 50
        print("[INFO] Eye threshold initialized successfully!")
        print("[INFO] Mouth threshold initialized successfully!")
        print("[INFO] Eye Threshold {0}, Mouth Threshold {1}".format(round(self.eye_threshold,3),round(self.mouth_threshold,3)))
        print("[INFO] Start Fatigue Detecting!")
        self.v_timer.start(20)

        self.v_timer.timeout.connect(self.show_pic)

        Ui_MainWindow.printf(window,"Load successfully, Start fatigue detection")
        window.statusbar.showMessage("Using camera...")

        self.output_data = []

    def init_threshold(self):
        success, frame = self.cap.read()

        ret, frame, _ = self.face_processor.frame_process(frame)
        eye, mouth = ret

        self.eye_threshold += eye
        self.mouth_threshold += mouth

    def show_pic(self):

        global EYE_AR_THRESH,EYE_AR_CONSEC_FRAMES,MAR_THRESH,MOUTH_AR_CONSEC_FRAMES,COUNTER,TOTAL,mCOUNTER,mTOTAL,ActionCOUNTER,Roll
        global hTOTAL,hmTOTAL
        global count
        global Gaze_Vectors_X,Gaze_Vectors_Y

        success, frame = self.cap.read()

        if success:
            count += 1
            if save_frame:
                save_path = os.path.join(image_path, "%05d.jpg" % (count))
                cv2.imwrite(save_path, frame)

            ret,frame, gaze_vector = self.face_processor.frame_process(frame)
            eye,mouth = ret

            window.label_13.setText("Gaze X: {}".format(gaze_vector[0]))
            window.label_14.setText("Gaze Y: {}".format(gaze_vector[1]))

            Gaze_Vectors_X.append(gaze_vector[0])
            Gaze_Vectors_Y.append(gaze_vector[1])

            if eye < (self.eye_threshold * 0.85):
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    hTOTAL += 1
                    window.label_3.setText("Blink Count:" + str(hTOTAL) )
                    COUNTER = 0
            window.label_10.setText("Blink Rate:" + str(round(TOTAL / count, 2)))

            if mouth > (self.mouth_threshold * 2):
                mCOUNTER += 1
            else:
                if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:
                    mTOTAL += 1
                    hmTOTAL += 1
                    window.label_4.setText("Yawn Count:" + str(hmTOTAL))
                    mCOUNTER = 0

            window.label_9.setText("Yawn Rate:" + str(round(mTOTAL / count, 2)))

            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            window.label.setPixmap(QPixmap.fromImage(showImage))

            Roll += 1

            if Roll == self.time_stamp:
                std_gaze_x = round(np.std(np.asarray(Gaze_Vectors_X)).item(),2)
                std_gaze_y = round(np.std(np.asarray(Gaze_Vectors_Y)).item(),2)

                if self.time_number == 0:
                    self.output_data.append(("Time_Second","Blink_Count","Yawn_Count","Gaze_X","Gaze_Y"))

                if output_path != "":
                    self.output_data.append(((self.time_number+1)*time_step, TOTAL, mTOTAL,std_gaze_x,std_gaze_y))

                Ui_MainWindow.printf(window, "Frame {} finished, Restart fatigue detection".format(self.time_stamp))

                window.label_3.setText("Blink Count:" + str(hTOTAL))
                window.label_10.setText("Blink Rate:" + str(round(hTOTAL / count, 2)))

                window.label_4.setText("Yawn Count:" + str(hmTOTAL))
                window.label_9.setText("Yawn Rate:" + str(round(hmTOTAL / count, 2)))

                Roll = 0

                TOTAL = 0
                mTOTAL = 0

                self.time_number += 1
                Gaze_Vectors_Y = []
                Gaze_Vectors_X = []

                output_file = codecs.open(anno_file, 'w', 'gbk')
                writer = csv.writer(output_file)
                for data in self.output_data:
                    writer.writerow(data)
                output_file.close()

class VideoDrivingBehaviorDetection(object):
    def __init__(self,args):
        self.init_nums = 50

        self.face_processor = FaceProcessor()
        self.eye_threshold = 0
        self.mouth_threshold = 0

        self.frame_count = 0
        self.period_frame_count = 0

        self.eye_count = 0
        self.blink_count = 0
        self.history_blink_count = 0

        self.mouth_count = 0
        self.yawn_count = 0
        self.history_yawn_count = 0

        self.args = args

        self.cap = cv2.VideoCapture(args.input_video)

        self.success_init = 0
        for _ in range(self.init_nums):
            self.init_threshold()

        self.eye_threshold /= self.success_init
        self.mouth_threshold /= self.success_init
        print("[INFO] Eye threshold initialized successfully!")
        print("[INFO] Mouth threshold initialized successfully!")
        print("[INFO] Eye Threshold {0}, Mouth Threshold {1}".format(round(self.eye_threshold, 3),
                                                              round(self.mouth_threshold, 3)))
        print("[INFO] Start Fatigue Detecting!")

        self.output_data = []
        self.time_number = 0

    def init_threshold(self):
        success, frame = self.cap.read()

        if not success:
            print("[ERROR] Total frame count is less than {}, please input longer video!".format(self.init_nums))
            sys.exit(-1)
        try:
            ret, frame, _ = self.face_processor.video_frame_process(frame)
            eye, mouth = ret

            self.eye_threshold += eye
            self.mouth_threshold += mouth
            self.success_init += 1
        except:
            pass

    def analysis_video(self):
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_step = fps * time_step
        frame_nums = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.gaze_vectors_x = []
        self.gaze_vectors_y = []

        with tqdm(range(frame_nums-self.init_nums),leave=False) as Frames:
            for _ in Frames:
                _, frame = self.cap.read()

                gaze_vector = [10, 10]
                frame_copy = copy.deepcopy(frame)

                try:
                    ret, frame, gaze_vector = self.face_processor.video_frame_process(frame)
                    frame_copy = copy.deepcopy(frame)

                    self.gaze_vectors_x.append(gaze_vector[0])
                    self.gaze_vectors_y.append(gaze_vector[1])

                    eye, mouth = ret
                    self.frame_count += 1

                    frame_copy = cv2.putText(frame_copy, "Frame: {}".format(self.frame_count), (0, 20),
                                             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255),
                                             thickness=2)

                    if eye < (self.eye_threshold * 0.85):
                        self.eye_count += 1
                    else:
                        if self.eye_count >= EYE_AR_CONSEC_FRAMES:
                            self.blink_count += 1
                            self.history_blink_count += 1
                            self.eye_count = 0

                    if mouth > (self.mouth_threshold * 1.6):
                        self.mouth_count += 1
                    else:
                        if self.mouth_count >= MOUTH_AR_CONSEC_FRAMES:
                            self.yawn_count += 1
                            self.history_yawn_count += 1
                            self.mouth_count = 0

                    self.period_frame_count += 1

                    if self.period_frame_count == self.frame_step:
                        std_gaze_x = round(np.std(np.asarray(self.gaze_vectors_x)).item(), 2)
                        std_gaze_y = round(np.std(np.asarray(self.gaze_vectors_y)).item(), 2)

                        if self.time_number == 0:
                            self.output_data.append(("Time_Second", "Blink_Count", "Yawn_Count", "Gaze_X", "Gaze_Y"))

                        if output_path != "":
                            self.output_data.append(
                                ((self.time_number + 1) * time_step, self.blink_count, self.yawn_count, std_gaze_x,
                                 std_gaze_y))

                        self.period_frame_count = 0
                        self.blink_count = 0
                        self.yawn_count = 0

                        self.gaze_vectors_x = []
                        self.gaze_vectors_y = []
                        self.time_number += 1

                        output_file = codecs.open(anno_file, 'w', 'gbk')
                        writer = csv.writer(output_file)
                        for data in self.output_data:
                            writer.writerow(data)
                        output_file.close()
                except:
                    pass
        try:
            Frames.close()
        except:
            pass

            # show_image = self.draw_result(frame_copy, gaze_vector)
            # cv2.imshow("capture", show_image)
            # if cv2.waitKey(100) & 0xFF == ord('q'):
            #     break
        print("[FINISHED] Analysis video successfully!")

    def draw_result(self, image, gaze_vector):
        image = cv2.putText(image, "Blink Count: {}/{}".format(self.history_blink_count,round(self.history_blink_count/self.frame_count,2)),
                            (0, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), thickness=2)
        image = cv2.putText(image, "Yawn Count: {}/{}".format(self.history_yawn_count,round(self.history_yawn_count/self.frame_count,2)),
                            (0, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0, 255, 255), thickness=2)
        image = cv2.putText(image, "Gaze X: {}  Gaze Y: {}".format(gaze_vector[0],gaze_vector[1]),
                            (0, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0, 255, 255), thickness=2)
        return image

def CamConfig_init():
    window.f_type = CamConfig()

def save_result_video(image_save_root, video_save_path,fps):
    files = os.listdir(image_save_root)
    num = len(files)
    if num == 0:
        print("No image in image root!")
        sys.exit(-1)
    file = files[0]
    image = cv2.imread(os.path.join(image_save_root,file))
    h,w,_ = image.shape

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, (w,h))

    for i in range(num):
        image_path = os.path.join(image_save_root,"%05d.jpg"%(i))
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    video_writer.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Driving behavior detection')
    parser.add_argument('-o','--output_path', default = "./output", type=str, help='store detection result')
    parser.add_argument('-i','--input_video_dir',default="",type=str,help='input video path')
    parser.add_argument('-s', '--save_frame', action='store_true', help="save frames or not")
    parser.add_argument('-t', '--time_step', default=1, type=int, help="interval per detection")
    args = parser.parse_args()

    output_path = args.output_path
    save_frame = args.save_frame
    time_step = args.time_step
    input_video_dir = args.input_video_dir

    if output_path != "":
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    if input_video_dir != "":
        files = os.listdir(input_video_dir)
        for file in tqdm(files):
            file_path = os.path.join(input_video_dir,file)
            video_name = os.path.basename(file_path)
            print()
            print("Process Video:{} Now".format(video_name))
            video_name = Path(video_name).stem
            anno_file = os.path.join(output_path, "{}.csv".format(video_name))
            args.anno_file = anno_file
            args.input_video = file_path
            detector = VideoDrivingBehaviorDetection(args)
            detector.analysis_video()
    else:
        anno_file = os.path.join(output_path, "anno.csv")
        if save_frame:
            image_path = os.path.join(output_path, "frames")
            if not os.path.exists(image_path):
                os.makedirs(image_path)

        app = QtWidgets.QApplication(sys.argv)
        window = MainWindow()
        window.window_init()
        window.show()
        sys.exit(app.exec_())