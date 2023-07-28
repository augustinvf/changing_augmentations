from typing import Union, Type, List

import torch
import random

# Initialize list to follow augmentation changes

def initialize_power_list(nb_classes: int, mini: int, maxi: int, 
                          powers_for_every_class: List[Union[int, float]]=None):
    if powers_for_every_class:
        power_list = [powers_for_every_class for _ in range (nb_classes)]
    else :
        power_list = [random.randint(mini, maxi) for _ in range (nb_classes)]
    return power_list

def initialize_operation_list(nb_classes: int, nb_augmentations: int, nb_same_time_operations: int, ops_for_every_class: list=None):
    if ops_for_every_class:
        operation_list = [ops_for_every_class for _ in range (nb_classes)]
    else :
        operation_list = [[random.randint(0, nb_augmentations-1) for _ in range(nb_same_time_operations)] for _ in range (nb_classes)]
    return operation_list

# Functions to change the augmentations

def compute_new_augmentations(nb_classes, power_list, ressemblance_matrix, threshold):
    for label in range(nb_classes):
        diff = evaluation_criterion(label, ressemblance_matrix)
        adjust_powers(diff, threshold, label, power_list)

def evaluation_criterion(label, ressemblance_matrix):
    """
    provides the comparison criterion for power adjustment
    """
    diff = 0
    maxi = torch.argmax(ressemblance_matrix[label,:])
    proba_maxi = ressemblance_matrix[label,maxi]
    diff += proba_maxi
    ressemblance_matrix[label,maxi] = 0
    snd_maxi = torch.argmax(ressemblance_matrix[label,:])
    poba_snd_maxi = ressemblance_matrix[label,snd_maxi]
    diff = (diff-poba_snd_maxi).item()
    return diff

def adjust_powers(criterion, threshold, label, power_list):
    if criterion > threshold and criterion > 0:
        change_power_list(power_list, label, 1)

def change_power_list(power_list, label, value):
    power_list[label] += value

# Applying the augmentations ie changing the attributs of the transformation to make the changes effective

def update_new_augmentations(self_supervised_augmentations, power_list: list):
    self_supervised_augmentations.update_power_list(power_list)
