import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torchvision import transforms

class EMNISTCSV(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path).values
        self.images = self.data[:, 1:].reshape(-1, 28, 28).astype(np.uint8)
        self.labels = self.data[:, 0].astype(np.int64) - 1  # Convert 1-26 to 0-25
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # EMNIST images are rotated - fix that by transposing
        image = np.transpose(image, (1, 0))  # Transpose to correct orientation

        if self.transform:
            image = self.transform(image)

        return image, label
