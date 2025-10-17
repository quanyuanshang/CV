"""
This module contains a custom dataset class for loading images and labels from a directory in the CS172 assignment 1.
The ImageDataset class is a subclass of the torch.utils.data.Dataset class, which is a PyTorch class for creating custom datasets.
This class doesn't need to store the images and labels in memory, it just needs to store the path list of the images.
This class is passed to the DataLoader class to load the images and labels in batches.
"""

import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ImageDataset(Dataset):
    """
    A custom dataset class for loading images and labels.
    """

    def __init__(self, imgdir, transform=None):
        """
        Args:
            imgdir (str): The directory where the images are stored, the labels are the name of the image.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.imgdir = imgdir
        self.transform = transform
        self.imgs = os.listdir(imgdir)
        self.resize = transforms.Resize((227, 227))

    def __len__(self):
        # ====================== TO DO START ========================
        # Return the number of images in the dataset
        # ===========================================================
        return len(self.imgs)
        # ====================== TO DO END ==========================

    def __getitem__(self, idx):
        # Load the image and label at the given index
        img_path = os.path.join(self.imgdir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.resize(img)

        # ====================== TO DO START ========================
        # return img and label, which should be a np.array or torch.tensor
        # label should be in one-hot format, the shape should be (5, 10)
        # Remeber Pillow Image shape is (H, W, C) and torch need (C, H, W)
        # Remeber Pillow Image type is uint8 ans torch need float32
        # Apply the transform on img if it exists
        # label (from list `self.imgs`) is like '12345#2.png'
        # ===========================================================
        if self.transform:
            img = self.transform(img)
        else:
            img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            img = img.transpose(2, 0, 1)
        filename = self.imgs[idx]
        label_str = filename.split('#')[0]
# filename = '12345#2.png'
# parts = filename.split('#')
# 结果：parts = ['12345', '2.png']
        label=np.zeros((5,10),dtype=np.float32)
        for i, digit_char in enumerate(label_str):
            label[i, int(digit_char)] = 1.0

        # ====================== TO DO END ==========================

        return img, label
    
    def get_samples(self, idx=0):
        assert idx < self.__len__()

        img_path = os.path.join(self.imgdir, self.imgs[idx])
        image = Image.open(img_path).convert("RGB")
        img, label = self.__getitem__(idx)

        return image, img, label
