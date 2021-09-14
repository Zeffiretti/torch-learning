from torch.utils.data import Dataset
from PIL import Image
import os
import cv2
from torch.utils.tensorboard import SummaryWriter


class ants(Dataset):
    def __init__(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = current_path + "/../dataset/hymenoptera_data/train"
        self.label_dir = "ants"
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)
