import onnxruntime
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import dlib
import cv2
import time
from cv2 import dnn
from gaze_estimation.gazenet import GazeNet
import torch

# init face detector from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('weights/shape_predictor_68_face_landmarks.dat')
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

def eye_aspect_ratio(eye):
    # Calculate EAR value
    # Divide the width of the eye by the length. If you close your eyes, this value will be smaller
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    # Calculate Mouth value
    # Divide the width of the mouth by the length. If you open your eyes, this value will be bigger
    A = np.linalg.norm(mouth[2] - mouth[10])
    B = np.linalg.norm(mouth[4] - mouth[8])
    C = np.linalg.norm(mouth[0] - mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

class FaceProcessor(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_bin = "./weights/opencv_face_detector_uint8.pb"
        config_text = "./weights/opencv_face_detector.pbtxt"
        gaze_model_pth = "./weights/gazenet.pth"

        # load tensorflow model
        self.face_net = cv2.dnn.readNetFromTensorflow(model_bin, config=config_text)
        print("[INFO] Loading facial detector successfully!")

        onnx_model_path = "./weights/face_align.onnx"
        self.session = onnxruntime.InferenceSession(onnx_model_path, None)
        # get the name of the first input of the model
        self.input_name = self.session.get_inputs()[0].name

        print("[INFO] Loading facial landmark predictor successfully!")

        # load gaze_net
        gaze_checkpoint = torch.load(gaze_model_pth, map_location=self.device)
        self.gaze_net = GazeNet(self.device)
        self.gaze_net.load_state_dict(gaze_checkpoint)
        self.gaze_net.eval()

        print("[INFO] Loading gaze estimation model successfully!")


    def eye_aspect_ratio(self, eye):
        # Calculate EAR value
        # Divide the width of the eye by the length. If you close your eyes, this value will be smaller
        A = dist.euclidean(eye[1], eye[7])
        B = dist.euclidean(eye[2], eye[6])
        C = dist.euclidean(eye[3], eye[5])
        W = dist.euclidean(eye[0], eye[4])
        ear = (A + B + C) / (3.0 * W)
        return ear

    def mouth_aspect_ratio(self, mouth):
        # Calculate Mouth value
        # Divide the width of the mouth by the length. If you open your eyes, this value will be bigger
        A = np.linalg.norm(mouth[1] - mouth[11])
        B = np.linalg.norm(mouth[2] - mouth[10])
        C = np.linalg.norm(mouth[3] - mouth[9])
        D = np.linalg.norm(mouth[4] - mouth[8])
        E = np.linalg.norm(mouth[5] - mouth[7])
        W = np.linalg.norm(mouth[0] - mouth[6])
        mar = (A + B + C + D +E) / (5.0 * W)
        return mar

    def face_detect(self, frame):
        height = frame.shape[0]
        width = frame.shape[1]

        # detect face from image
        blobImage = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        self.face_net.setInput(blobImage)
        output = self.face_net.forward()

        x, y, w, h = None, None, None, None

        # Keep the face with high confidence
        for detection in output[0, 0, :1, :]:
            score = float(detection[2])
            # confidence threshold is 0.2
            if score > 0.2:
                left = detection[3] * width
                top = detection[4] * height
                right = detection[5] * width
                bottom = detection[6] * height
                x, y, w, h = left, top, right - left, bottom - top
            else:
                return None, None, None, None
        # return x,y,w,h of face
        # x,y is left-top point of face
        x, y, w, h = list(map(int, [x, y, w, h]))
        return x, y, w, h

    def pfld_detect(self,frame):
        x, y, w, h = self.face_detect(frame)
        if x is None:
            return frame

        # transfer face to dlib rectangle
        rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))

        # predict face landmark
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmarks = predictor(gray, rect)

        landmarks = face_utils.shape_to_np(landmarks)

        # Get eyes and mouth region landmark
        leftEye = landmarks[lStart:lEnd]
        rightEye = landmarks[rStart:rEnd]
        mouth = landmarks[mStart:mEnd]

        # calculate eyeEAR and mouthAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        eyear = (leftEAR + rightEAR) / 2.0
        mouthar = mouth_aspect_ratio(mouth)

        frame_copy = frame.copy().astype(np.uint8)

        # visualize the landmarks in eye and mouth areas
        for (x, y) in leftEye:
            cv2.circle(frame_copy, (int(x), int(y)), 1, (0, 255, 0), -1)
        for (x, y) in rightEye:
            cv2.circle(frame_copy, (int(x), int(y)), 1, (0, 255, 0), -1)
        for (x, y) in mouth:
            cv2.circle(frame_copy, (int(x), int(y)), 1, (255, 0, 0), -1)

        # calculate gaze vector by model and landmark
        frame_copy,gaze_vector= self.get_gaze(landmarks, frame, frame_copy)
        return frame_copy,eyear,mouthar,gaze_vector

    def get_gaze(self, landmarks, frame, display):
        # Calculate eye center point
        left_eye_center = (landmarks[37,:] + landmarks[38,:] + landmarks[40,:] + landmarks[41,:])/4
        right_eye_center = (landmarks[43, :] + landmarks[44,:] + landmarks[46,:] + landmarks[47,:])/4
        # Calculate mouth corner point
        left_mouth_corner = landmarks[48,:]
        right_mouth_corner = landmarks[64,:]

        nose_center = landmarks[33,:]

        # normalize landmark for subsequent calculation
        normalize_landmarks = [left_eye_center[0],right_eye_center[0],nose_center[0],left_mouth_corner[0],right_mouth_corner[0],
                               left_eye_center[1],right_eye_center[1],nose_center[1],left_mouth_corner[1],right_mouth_corner[1],]

        frame_copy = frame.copy()
        frame_copy = frame_copy[:, :, ::-1]
        frame_copy = cv2.flip(frame_copy, 1)
        img_h, img_w, _ = np.shape(frame_copy)

        # Align face according to landmark
        face, gaze_origin, M = self.normalize_face(normalize_landmarks, frame_copy)

        # forward gaze net
        with torch.no_grad():
            gaze = self.gaze_net.get_gaze(face)
            gaze = gaze[0].data.cpu()

        # draw gaze for display
        display,gaze_vector = self.draw_gaze(display, left_eye_center, gaze, color=(0, 0, 255), thickness=1)
        display,gaze_vector = self.draw_gaze(display, right_eye_center, gaze, color=(0, 0, 255), thickness=1)
        return display,gaze_vector

    def draw_gaze(self, image_in, eye_pos, pitchyaw, length=100, thickness=1, color=(0, 0, 255)):
        # Draw the gaze vector as an arrow in image
        image_out = image_in
        if len(image_out.shape) == 2 or image_out.shape[2] == 1:
            image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)

        # Gets the tail of the arrow
        dx = -length * np.sin(pitchyaw[1])
        dy = -length * np.sin(pitchyaw[0])
        cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                        tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                        thickness, cv2.LINE_AA, tipLength=0.5)
        dx = round(dx.item(),2)
        dy = round(dy.item(),2)
        return image_out,(dx,dy)

    def normalize_face(self, landmarks, frame):
        # We need to use landmark to align face
        left_eye_coord = (0.70, 0.35)

        lcenter = tuple([landmarks[0], landmarks[5]])
        rcenter = tuple([landmarks[1], landmarks[6]])

        gaze_origin = (int((lcenter[0] + rcenter[0]) / 2), int((lcenter[1] + rcenter[1]) / 2))

        dY = rcenter[1] - lcenter[1]
        dX = rcenter[0] - lcenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        right_eye_x = 1.0 - left_eye_coord[0]

        dist = np.sqrt((dX ** 2) + (dY ** 2))
        new_dist = (right_eye_x - left_eye_coord[0])
        new_dist *= 112
        scale = new_dist / dist

        # Calculate transformation matrix
        M = cv2.getRotationMatrix2D(gaze_origin, angle, scale)

        tX = 112 * 0.5
        tY = 112 * left_eye_coord[1]
        M[0, 2] += (tX - gaze_origin[0])
        M[1, 2] += (tY - gaze_origin[1])

        # Computing affine transformation from transformation matrix
        face = cv2.warpAffine(frame, M, (112, 112),
                              flags=cv2.INTER_CUBIC)
        return face, gaze_origin, M

    def frame_process(self, frame):
        ret = []

        tstart = time.time()

        # get process result
        frame, eye, mouth, gaze_vector = self.pfld_detect(frame)

        ret.append(round(eye, 3))
        ret.append(round(mouth, 3))

        # show process fps
        tend = time.time()
        fps = 1 / (tend - tstart)
        fps = "%.2f fps" % fps
        cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

        return ret, frame, gaze_vector

    def video_frame_process(self, frame):
        ret = []
        # this function is for video
        frame, eye, mouth, gaze_vector = self.pfld_detect(frame)

        ret.append(round(eye, 3))
        ret.append(round(mouth, 3))

        return ret, frame, gaze_vector

