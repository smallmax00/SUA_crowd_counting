from torch.utils import data
from PIL import Image
import numpy as np
import h5py
import os
from .transforms_semantic import Transforms
import glob
from torchvision.transforms import functional
import cv2


class Dataset(data.Dataset):
    def __init__(self, data_path, dataset, is_train):
        self.is_train = is_train
        self.dataset = dataset
        if is_train:
            is_train = 'train_data'
        else:
            is_train = 'test_data'
        if dataset == 'SHA':
            dataset = 'part_A_final'
        elif dataset == 'SHB':
            dataset = 'part_B_final'

        self.image_list = glob.glob(os.path.join(data_path, dataset, is_train, 'images', '*.jpg'))
        self.label_list = glob.glob(os.path.join(data_path, dataset, is_train, 'new_data', '*.h5'))
        self.image_list.sort()
        self.label_list.sort()

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert('RGB')
        label = h5py.File(self.label_list[index], 'r')
        density = np.array(label['density'], dtype=np.float32)
        attention = np.array(label['attention'], dtype=np.float32)
        gt = np.array(label['gt'], dtype=np.float32)
        trans = Transforms((0.8, 1.2), (128, 128), 1, (0.5, 1.5), self.dataset)
        if self.is_train:
            image, density, attention = trans(image, density, attention)
            return image, density, attention
        else:
            height, width = image.size[1], image.size[0]
            height = round(height / 16) * 16
            width = round(width / 16) * 16
            image = image.resize((width, height), Image.BILINEAR)
            image = functional.to_tensor(image)
            image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            density = cv2.resize(density.astype('float32'), (width, height), interpolation=cv2.INTER_LINEAR)
            attention = cv2.resize(attention.astype('float32'), (width, height), interpolation=cv2.INTER_LINEAR)

            return image, gt, density, attention

    def __len__(self):
        return len(self.image_list)

