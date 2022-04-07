import os
import json
import random

from PIL import Image
from tqdm import tqdm

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def image_loader(path):
    image = Image.open(path).convert('RGB')
    # resize all image to 224x224, Because the CNN network we use can only receive fixed size input
    image = image.resize((224,224))
    return image

class DCNDataset(Dataset):
    def __init__(self, json_path):
        # load data from json file
        data = open(json_path).readlines()
        self.result = []
        for line in tqdm(data):
            line = json.loads(line)
            self.result.append(line)
        # shuffle the data
        random.shuffle(self.result)

        # init transform for image
        mean = (0.5, 0.5, 0.5)
        stds= (0.5, 0.5, 0.5)
        self.transform=transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean,stds)])

    def __getitem__(self, index):
        # load sparse feature
        json_data = self.result[index]
        blink = json_data.get("blink")
        yawn = json_data.get("yawn")
        gaze_x = json_data.get("gaze_x")
        gaze_y = json_data.get("gaze_y")
        heart_rate = json_data.get("heart")
        # load image
        images = self.load_image(json_data.get("image_path"))
        label = json_data.get("label")-1
        # return data tuple
        return (blink,yawn,gaze_x,gaze_y,heart_rate,images),label

    def load_image(self, image_name):
        # read image
        data_root = r"D:\Chrome Download\CSCNN_dataset"
        image_root = os.path.join(data_root, "processed_frames")
        image_path = os.path.join(image_root,image_name)
        image = image_loader(image_path)
        # transform image by torchvision
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.result)

if __name__ == '__main__':
    # Test
    train_data = DCNDataset(json_path=r"D:\Chrome Download\CSCNN_dataset\train.json")
    data_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        pass
    pass