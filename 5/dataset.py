import os
import random
import shutil
import warnings
from glob import glob

import PIL
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class VidDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, input_size, mode='train', transforms=None):
        self.mode = mode

        self.class_list = ['bird', 'car', 'dog', 'lizard', 'turtle']
        self.data = []

        for c in range(len(self.class_list)):
            with open(os.path.join(img_path, self.class_list[c] + '_gt.txt')) as f:
                label = f.readlines()

            class_pic = os.listdir(os.path.join(img_path, self.class_list[c]))
            class_pic.sort()

            if mode == 'train':
                class_pic = class_pic[:150]
                label = label[:150]
            else:
                class_pic = class_pic[150:180]
                label = label[150:180]

            for p in range(len(class_pic)):
                p_box = label[p][:-1].split(' ')[1:]
                p_box = torch.tensor([(float(i) - 64) / 64 for i in p_box])
                p_data = [os.path.join(os.path.join(img_path, self.class_list[c], class_pic[p])),
                          c, p_box]
                self.data.append(p_data)

        if transforms is None:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            self.transforms = torchvision.transforms.Compose([
                # torchvision.transforms.Resize(size=input_size),  # 缩放
                # Resize(size=input_size),  # 等比填充缩放
                # torchvision.transforms.RandomCrop(size=input_size),
                # torchvision.transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),
                torchvision.transforms.RandomHorizontalFlip(),
                # RandomGaussianBlur(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = PIL.Image.open(self.data[idx][0])
        img = self.transforms(img)

        sample = {"image": img, "class": self.data[idx][1],
                  "box": self.data[idx][2], "path": self.data[idx][0]}
        return sample
