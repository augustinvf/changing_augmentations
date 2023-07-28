from typing import Optional, Tuple, Union

import torchvision.transforms as T

from lightly.transforms.rotation import random_rotation_transform
from lightly.transforms.gaussian_blur import GaussianBlur

class SimCLR() :
    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 0.5,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.8,
        cj_hue: float = 0.2,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: int = 15,
        power_list = [],
    ) :
        self.cj_strength = cj_strength
        self.cj_bright = cj_bright
        self.cj_contrast = cj_contrast
        self.cj_sat = cj_sat
        self.cj_hue = cj_hue
        self.rr_prob = rr_prob
        self.hf_prob = hf_prob
        self.vf_prob = vf_prob
        self.cj_prob = cj_prob
        self.random_gray_scale = random_gray_scale
        self.kernel_size = kernel_size
        self.gaussian_blur = gaussian_blur
        self.power_list = power_list    
        self.min_scale = min_scale
        self.rr_degrees = rr_degrees
        self.sigmas = sigmas

    def __call__(self, image, label) :

        power = self.power_list[label] / 5

        color_jitter = T.ColorJitter(
            brightness=self.cj_strength * self.cj_bright * power,
            contrast=self.cj_strength * self.cj_contrast * power,
            saturation=self.cj_strength * self.cj_sat * power,
            hue=self.cj_strength * self.cj_hue * power,
        )
        print(self.cj_strength * self.cj_hue * power)
        transform = [
            T.RandomResizedCrop(size=32, scale=(self.min_scale, 1.0)),
            random_rotation_transform(rr_prob=self.rr_prob, rr_degrees=self.rr_degrees*power),
            T.RandomHorizontalFlip(p=self.hf_prob),
            T.RandomVerticalFlip(p=self.vf_prob),
            T.RandomApply([color_jitter], p=self.cj_prob),
            T.RandomGrayscale(p=self.random_gray_scale),
            GaussianBlur(kernel_size=self.kernel_size, sigmas=[self.sigmas[0]*power, self.sigmas[1]*power], prob=self.gaussian_blur),
            T.ToTensor(),
            T.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2470, 0.2435, 0.2616])
        ]

        self.transform = T.Compose(transform)
        return [self.transform(image), self.transform(image)]
    
    def update_power_list(self, power_list: list):
        self.power_list = power_list