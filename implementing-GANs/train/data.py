import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
import torch
import time


class ImageLoader(Dataset):
    def __init__(self, data_path, split, file_path='data.npy', flatten=False):
        file_path = os.path.join(data_path, file_path)
        self.dataset = np.load(file_path)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        transform = [transforms.ToTensor(), self.normalize]
        self.preprocess = transforms.Compose(transform)
        self.flatten = flatten

        self.split = split
        self.images = self.dataset # shape 10000, 128, 128, 3

        # if file_path == os.path.join(data_path, 'data.npz'):
        #     self.images = self.images[:50000]#np.reshape(np.array(tmp['arr_0']), (50000, 32, 32, 3))
        #         #self.images /= np.sqrt(2)
        #         self.labels = self.labels[:50000]
        #     if split == 'test':
        #         self.images = self.images[50000:]
        #         self.labels = self.labels[50000:]

        self.process = True if len(self.images.shape) > 2 else False

    def __getitem__(self, index):
        image = self.images[index]# + np.reshape(self.images2[index], (32, 32, 3))
        #image /= np.sqrt(2)

        if self.process:
            img_tensor = self.preprocess(Image.fromarray(image.astype(np.int32), 'RGB'))
        else:
            img_tensor = torch.from_numpy(image)
            img_tensor = img_tensor.float()

        if self.flatten:
            if len(img_tensor.size()) > 2:
                batch_size = img_tensor.size(0)
                img_tensor = img_tensor.view(128 * 128 * 3)
        else:
            if len(img_tensor.size()) == 2:
                img_tensor = img_tensor.view(128, 128, 3)

        return img_tensor

    def __len__(self):
        return self.images.shape[0]
