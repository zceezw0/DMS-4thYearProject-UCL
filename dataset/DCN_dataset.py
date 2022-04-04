from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import cv2
import os
import json
import random
import torch
import numpy as np

def image_loader(path):
    image = Image.open(path).convert('RGB')
    image = image.resize((224,224))
    return image

class DCNDataset(Dataset):
    def __init__(self, json_path):
        data = open(json_path).readlines()
        self.result = []
        for line in tqdm(data):
            line = json.loads(line)
            self.result.append(line)
        random.shuffle(self.result)

        mean = (0.5, 0.5, 0.5)
        stds= (0.5, 0.5, 0.5)
        self.transform=transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean,stds)])

    def __getitem__(self, index):
        json_data = self.result[index]
        data = {}
        #torch.tensor(np.array([json_data.get("yawn")]), dtype=torch.long)
        blink = json_data.get("blink")
        yawn = json_data.get("yawn")
        gaze_x = json_data.get("gaze_x")
        gaze_y = json_data.get("gaze_y")
        heart_rate = json_data.get("heart")
        images = self.load_image(json_data.get("image_path"))
        label = json_data.get("label")-1
        return (blink,yawn,gaze_x,gaze_y,heart_rate,images),label

    def load_image(self, image_name):
        data_root = r"D:\Chrome Download\CSCNN_dataset"
        image_root = os.path.join(data_root, "processed_frames")
        image_path = os.path.join(image_root,image_name)
        image = image_loader(image_path)
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.result)

if __name__ == '__main__':
    train_data = DCNDataset(json_path=r"D:\Chrome Download\CSCNN_dataset\train.json")
    data_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        pass
    pass