from PIL import Image
import numpy as np
import cv2
import random
from torchvision.transforms import functional


class Transforms(object):
    def __init__(self, scale, crop, stride, gamma, dataset):
        self.scale = scale
        self.crop = crop
        self.stride = stride
        self.gamma = gamma
        self.dataset = dataset

    def __call__(self, image, density, attention):
        # random resize
        height, width = image.size[1], image.size[0]

        # random crop
        h, w = self.crop[0], self.crop[1]
        dh = random.randint(0, height - h)
        dw = random.randint(0, width - w)
        image = image.crop((dw, dh, dw + w, dh + h))
        density = density[dh:dh + h, dw:dw + w]
        attention = attention[dh:dh + h, dw:dw + w]


        # random flip
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            density = density[:, ::-1]
            attention = attention[:, ::-1]


        # random gamma
        if random.random() < 0.3:
            gamma = random.uniform(self.gamma[0], self.gamma[1])
            image = functional.adjust_gamma(image, gamma)

        # random to gray
        if self.dataset == 'SHA':
            if random.random() < 0.1:
                image = functional.to_grayscale(image, num_output_channels=3)

        image = functional.to_tensor(image)
        image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])



        density = cv2.resize(density, (density.shape[1] // self.stride, density.shape[0] // self.stride),
                             interpolation=cv2.INTER_LINEAR) * self.stride * self.stride

        attention = cv2.resize(attention, (attention.shape[1] // self.stride, attention.shape[0] // self.stride),
                               interpolation=cv2.INTER_LINEAR)

        density = np.reshape(density, [1, density.shape[0], density.shape[1]])
        attention = np.reshape(attention, [1, attention.shape[0], attention.shape[1]])


        return image, density, attention
