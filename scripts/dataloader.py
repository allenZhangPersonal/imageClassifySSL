import os
from PIL import Image
import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform, rotNet=False, rotDegree=0):
        r"""
        Args:
            root: Location of the dataset folder
            split: The split you want to used, it should be one of labeled, unlabeled or validation
            transform: The transformation you want to apply to the images
            rotNet: Boolean indicating if you want to perform semi-supervised training on rotation network
            rotDegree: Rotation degree, we will usually create 4x the data through 0, 90, 180, and 270 degree rotations
        """
        self.split = split
        self.transform = transform

        self.image_dir = os.path.join(root, split)
        label_path = os.path.join(root, "{}_label_tensor.pt".format(split)) # Make sure labels for the images are under your root directory
        self.num_images = len(os.listdir(self.image_dir)) # Make sure only images are present in folder

        if rotNet:
            if rotDegree == 0:
                self.labels = torch.zeros(self.num_images, dtype=torch.long)
            elif rotDegree == 1:
                self.labels = torch.ones(self.num_images, dtype=torch.long)
            elif rotDegree == 2:
                self.labels = torch.ones(self.num_images, dtype=torch.long) + 1
            elif rotDegree == 3:
                self.labels = torch.ones(self.num_images, dtype=torch.long) + 2
            else: # Unlabeled
                self.labels = -1 * torch.ones(self.num_images, dtype=torch.long)
        else:
            if os.path.exists(label_path):
                self.labels = torch.load(label_path)
            else: # Unlabeled
                self.labels = -1 * torch.ones(self.num_images, dtype=torch.long)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx): # Iterator advances per call
        r"""
        Notes:
            Make sure to name your images from 0 to number of images - 1
            This way the labels from your .pt or .pth files will match directly with the image index
            Good for bookkeeping purposes
        """
        with open(os.path.join(self.image_dir, f"{idx}.png"), 'rb') as f:
            img = Image.open(f).convert('RGB')
            
        return self.transform(img), self.labels[idx] # Returns tuple for the transformed image and its corresponding label
