import os
import cv2
import sys
from tqdm import tqdm

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

    for i in tqdm(range(num)):
        image_path = os.path.join(image_save_root,"%05d.jpg"%(i))
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    video_writer.release()


if __name__ == '__main__':
    image_save_root = "./output/frames/"
    video_save_path = "./output/demo.avi"
    fps = 5

    if not os.path.exists(image_save_root):
        print("No image dir!")
        sys.exit(-1)

    save_result_video(image_save_root,video_save_path,fps)
