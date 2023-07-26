import wandb
wandb.login()

import torch
import torch.nn as nn

from lightly.models.modules import SimCLRProjectionHead
from lightly.loss import NTXentLoss

from model import Model
from dataloader import train_dataset_self_supervised, train_dataloader_self_supervised, train_dataloader_supervised, test_dataloader, batch_size
from training import self_supervised_training, supervised_training
from update_augmentations import initialize_power_list, initialize_operation_list
from update_augmentations import compute_new_augmentations, update_new_augmentations, check_operation_list
from augmentations import TransformForOneImage, len_augment_list
from eval import test_fct

wandb.init(
    project = "changing_augmentations_project",
    name = "run_1"
)

# tool initialization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nb_classes = 10
input_size_classifier = 512
projection_head = SimCLRProjectionHead(512, 512, 128)
nb_steps = len(train_dataloader_supervised)

nb_cycles = 10
nb_epochs_self_supervised_by_cycle = 2
nb_epochs_supervised_by_cycle = 1

# hyperparameters for augmentation updates

softmax = nn.Softmax(dim=0)
nb_augmentations = len_augment_list()
nb_same_time_operations = 2
power_list = initialize_power_list(nb_classes, nb_augmentations, 0, 30)
operation_list = initialize_operation_list(nb_classes, nb_augmentations, nb_same_time_operations)   # operations whose powers are currently adjusted
norm = 2
threshold = 0.3
adjustment = True
old_results = torch.tensor([0 for _ in range(nb_classes)])
states = [True for _ in range(nb_classes)]
cycle_min_for_adjustments = -1
cycle_max_for_adjustments = nb_steps * nb_epochs_self_supervised_by_cycle / 2
ressemblance_matrix = torch.rand((nb_classes, nb_classes), device=device)

# configuring the training dataset whose augmentations will change

self_supervised_augmentations = TransformForOneImage(power_list, operation_list)
train_dataset_self_supervised.update_self_supervised_augmentations(self_supervised_augmentations)

# hyperparameters for the model

model = Model(projection_head, input_size_classifier, nb_classes).to(device)

criterion_ss = NTXentLoss()
optimizer_ss = torch.optim.SGD(list(model.backbone.parameters()) + list(model.projection_head.parameters()), 0.3, momentum = 0.9, weight_decay=1e-6)
scheduler_ss = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ss, T_max=nb_cycles*nb_epochs_self_supervised_by_cycle, eta_min=0,
                                                           last_epoch=-1)

criterion_su = nn.CrossEntropyLoss()
optimizer_su = torch.optim.SGD(model.classifier.parameters(), lr = 0.1, momentum = 0.9, weight_decay=5e-4)
scheduler_su = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_su, T_max=nb_cycles*nb_epochs_supervised_by_cycle, eta_min=0,
                                                           last_epoch=-1)

# training

for cycle in range (nb_cycles) :
    print("debut cycle")
    for epochs in range(nb_epochs_self_supervised_by_cycle) :
        sum_loss_ss = self_supervised_training(device, model, train_dataloader_self_supervised, criterion_ss, optimizer_ss, scheduler_ss)
        wandb.log({"loss self-supervised": sum_loss_ss/nb_steps,
                   "learning rate self-supervised": scheduler_ss.get_last_lr()[0]
                })
    print("début supervised")
    for epochs in range(nb_epochs_supervised_by_cycle) :
        sum_loss_su, accuracy, r_matrix_not_normalized = supervised_training(device, model, train_dataloader_supervised, criterion_su, optimizer_su, scheduler_su, softmax, ressemblance_matrix)
        wandb.log({"loss supervised": sum_loss_su/nb_steps,
               "accuracy supervised": accuracy/(batch_size*nb_steps),
               "learning rate supervised": scheduler_su.get_last_lr()[0]
                })
        print("r_matrix", r_matrix_not_normalized/nb_classes)
    cycle_min_for_adjustments < cycle < cycle_max_for_adjustments
    if adjustment:
        print("je commence à ajuster les paramètres")
        compute_new_augmentations(nb_classes, power_list, operation_list, old_results, states, r_matrix_not_normalized/nb_classes, threshold, norm)
        update_new_augmentations(self_supervised_augmentations, power_list, operation_list)
        check_operation_list(nb_classes, states, nb_augmentations, operation_list)
        ressemblance_matrix.fill_(0)
    print(power_list)
    print(operation_list)

# test

test_accuracy = test_fct(device, model, test_dataloader)
wandb.log({"test_accuracy": test_accuracy})
