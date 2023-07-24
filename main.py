import wandb
wandb.login()

import torch
import torch.nn as nn

from lightly.models.modules import SimCLRProjectionHead
from lightly.loss import NTXentLoss

from model import Model
from dataloader import train_dataset_self_supervised, train_dataloader_self_supervised, train_dataloader_supervised, test_dataloader, batch_size
from training import self_supervised_training, supervised_training
from update_augmentations import compute_new_augmentations, update_powers, initialize_power_list, apply_new_augmentations
from transforms import MyTransformForOneImage
from eval import test_fct

wandb.init(
    project = "deep_learning_project_2",
    name = "Run_test_5"
)

# tool initialization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nb_classes = 10
input_size_classifier = 512
projection_head = SimCLRProjectionHead(512, 512, 128)
nb_steps = len(train_dataloader_supervised)

nb_cycles = 1
nb_epochs_self_supervised = 100
nb_epochs_supervised = 100

# hyperparameters for augmentation updates

softmax = nn.Softmax(dim=0)
class_transform = MyTransformForOneImage
power_list, nb_powers = initialize_power_list(nb_classes, [0.5, 0.5])
norm = 2
threshold = 0.3
current_power = 0
power_coefficient = 1
power_adjustment = 0
nb_adjustments = 0
nb_max_adjustments = nb_steps * nb_epochs_self_supervised / 2
adjustment = True

# hyperparameters for the model

model = Model(projection_head, input_size_classifier, nb_classes).to(device)

criterion_ss = NTXentLoss()
optimizer_ss = torch.optim.SGD(list(model.backbone.parameters()) + list(model.projection_head.parameters()), 0.3, momentum = 0.9, weight_decay=1e-6)
scheduler_ss = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ss, T_max=nb_cycles*nb_epochs_self_supervised, eta_min=0,
                                                           last_epoch=-1)

criterion_su = nn.CrossEntropyLoss()
optimizer_su = torch.optim.SGD(model.classifier.parameters(), lr = 0.1, momentum = 0.9, weight_decay=5e-4)
scheduler_su = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_su, T_max=nb_cycles*nb_epochs_supervised, eta_min=0,
                                                           last_epoch=-1)

# training

for cycles in range (nb_cycles) :
    for epochs in range(nb_epochs_self_supervised) :
        sum_loss_ss = self_supervised_training(device, model, train_dataloader_self_supervised, criterion_ss, optimizer_ss, scheduler_ss)
        wandb.log({"loss self-supervised": sum_loss_ss/nb_steps,
                   "learning rate self-supervised": scheduler_ss.get_last_lr()[0]
                })
    for epochs in range(nb_epochs_supervised) :
        sum_loss_su, accuracy, r_matrix = supervised_training(device, model, train_dataloader_supervised, criterion_su, optimizer_su, scheduler_su, nb_classes, softmax)
        wandb.log({"loss supervised": sum_loss_su/nb_steps,
               "accuracy supervised": accuracy/(batch_size*nb_steps),
               "learning rate supervised": scheduler_su.get_last_lr()[0]
                })
        if power_adjustment < nb_max_adjustments and adjustment:
            compute_new_augmentations(nb_classes, power_list, current_power, power_coefficient, r_matrix, threshold, norm)
            current_power, power_adjustment = update_powers(current_power, power_adjustment, nb_powers, nb_adjustments, nb_max_adjustments)
            apply_new_augmentations(train_dataset_self_supervised, class_transform, power_list)

# test

final_accuracy = test_fct(device, model, test_dataloader)
wandb.log({"final_accuracy": final_accuracy})
