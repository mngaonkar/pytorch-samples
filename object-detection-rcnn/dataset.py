import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "labels"))))

    def __getitem__(self, idx):
        # Load images and labels
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        label_path = os.path.join(self.root, "labels", self.labels[idx])

        img = Image.open(img_path).convert("RGB")
        boxes = []
        labels = []
        
        # Load bounding boxes and labels
        with open(label_path) as f:
            for line in f:
                components = line.strip().split()
                xmin = float(components[0])
                ymin = float(components[1])
                xmax = float(components[2])
                ymax = float(components[3])
                label = int(components[4])
                
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms:
            img = self.transforms(img)
        else:
            img = T.ToTensor()(img)

        return img, target

    def __len__(self):
        return len(self.imgs)
