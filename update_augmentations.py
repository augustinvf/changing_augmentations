from typing import Union, Type, List

import torch
import random

from augmentations import augment_list

# Initialize list to follow augmentation changes

def initialize_power_list(nb_classes: int, nb_augmentations: int, mini: int, maxi: int, 
                          val_for_every_power_and_class: int=None, 
                          powers_for_every_class: List[Union[int, float]]=None):
    if val_for_every_power_and_class:
        power_list = [val_for_every_power_and_class  for _ in range (nb_classes)]
    elif powers_for_every_class:
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

def compute_new_augmentations(nb_classes, power_list, operation_list, old_results, 
                              states, ressemblance_matrix, threshold, p=2):
    for label in range(nb_classes):
        diff = evaluation_criterion(label, ressemblance_matrix, p)
        has_changed = adjust_powers(diff, threshold, old_results, label, power_list, operation_list)
        old_results[label] = diff
        states[label] = has_changed

def evaluation_criterion(label, ressemblance_matrix, p=2):
    """
    provides the comparison criterion for power adjustment
    """
    maxi = torch.argmax(ressemblance_matrix[label,:])
    proba_maxi = ressemblance_matrix[label,maxi]
    ressemblance_matrix[label,maxi] = 0
    snd_maxi = torch.argmax(ressemblance_matrix[label,:])
    poba_snd_maxi = ressemblance_matrix[label,snd_maxi]
    diff = torch.norm((proba_maxi-poba_snd_maxi).float(), p=p).item()
    print("la classe", maxi, "et", snd_maxi, "se ressemblent et ont une diff de : ", diff)
    return diff

def adjust_powers(criterion, threshold, old_results, label, power_list, operation_list):
    has_changed = True
    if criterion > threshold :
        print("on change une puissance")
        change_power_list(power_list, label, operation_list, 2)
    else :
        if old_results[label] < threshold :
            has_changed = False
    return has_changed

def change_power_list(power_list, label, operation_list, value):
    if value < 0 :
        print("moins de puissance pour la classe", label)
        for power in operation_list[label] :
            power_list[label][power] = max(value + power_list[label][power], 0)
    else :
        print("plus de puissance pour la classe", label)
        for power in operation_list[label] :
            power_list[label][power] = min(value + power_list[label][power], 10)

# Applying the augmentations ie changing the attributs of the transformation to make the changes effective

def update_new_augmentations(self_supervised_augmentations, power_list: list, operation_list: list):
    self_supervised_augmentations.update_power_list(power_list)
    #self_supervised_augmentations.update_operation_list(operation_list)

# Looking at the state of the transformations : if a state has not changed, choose new one

def check_operation_list(nb_classes, states, nb_augmentations, operation_list):
    for label in range(nb_classes):
        if not states[label] :
            new_powers = random.sample(augment_list(), k=nb_augmentations)
            operation_list[label] = new_powers
