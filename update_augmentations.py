from typing import Union, Type

import torch

from transforms import MyTransformForOneImage

def initialize_power_list(nb_classes: int, powers: list[Union[float, int]]):
    nb_powers = len(powers)
    power_list = [powers for _ in range(nb_classes)]
    return power_list, nb_powers
    
def compute_new_augmentations(nb_classes, power_list, current_power, power_coefficient, ressemblance_matrix, threshold, p=2):
    for label in range(nb_classes):
        maxi = torch.argmax(ressemblance_matrix[label,:])
        ressemblance_matrix[label,maxi] = 0
        snd_maxi = torch.argmax(ressemblance_matrix[label,:])
        diff = torch.norm(maxi-snd_maxi, p=p)
        if threshold <= diff :
            power_list[label][current_power] *= power_coefficient * (1 + diff**2)

def apply_new_augmentations(dataset: Type[type], class_transform: Type[type], power_list: list):
     dataset.update_transform(MyTransformForOneImage)
     dataset.update_power_list(power_list)

def update_powers(current_power, power_adjustment, nb_powers, nb_adjustments):
        power_adjustment += 1
        if power_adjustment % nb_adjustments == 0 :
            current_power += 1
            if current_power == nb_powers :
                current_power = 0
        return current_power, power_adjustment
