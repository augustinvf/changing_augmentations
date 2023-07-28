import wandb
wandb.login()

import omegaconf

import torch
import torch.nn as nn

from lightly.models.modules import SimCLRProjectionHead
from lightly.loss import NTXentLoss

from model import Model
from dataloader import initialize_dataloader
from training import self_supervised_training, supervised_training
from update_augmentations import initialize_power_list
from update_augmentations import compute_new_augmentations, update_new_augmentations
from augmentations import len_augment_list
from sim_clr_transforms import SimCLR
from eval import test_fct

wandb.init(
    project = "changing_augmentations_project",
    name = "run_1"
)

# to load the hyperparameters

config = omegaconf.OmegaConf.load("config.yaml")

# data initialization

batch_size = config.batch_size
train_dataloader_self_supervised, train_dataloader_supervised, test_dataloader, train_dataset_self_supervised = initialize_dataloader(batch_size)

# tool initialization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nb_classes = config.nb_classes
input_size_classifier = config.input_size_classifier
projection_head = SimCLRProjectionHead(512, 512, 128)
nb_steps = len(train_dataloader_supervised)
confusion_matrix = config.confusion_matrix
nb_cycles = config.nb_cycles
nb_epochs_self_supervised_by_cycle = config.nb_epochs_self_supervised_by_cycle
nb_epochs_supervised_by_cycle = config.nb_epochs_supervised_by_cycle

# hyperparameters for augmentation updates

augmentation_adjustments=False

softmax = nn.Softmax(dim=1)
nb_augmentations = len_augment_list()
power_list = initialize_power_list(nb_classes, nb_augmentations, 1, 1)
threshold = config.threshold
ressemblance_matrix = torch.zeros((nb_classes, nb_classes), device=device)
nb_experiences_by_class = torch.zeros((1, nb_classes), device=device)

# configuring the training dataset whose augmentations will change

self_supervised_augmentations = SimCLR(power_list=power_list, gaussian_blur=config.gaussian_blur, min_scale=config.min_scale)
train_dataset_self_supervised.update_self_supervised_augmentations(self_supervised_augmentations)

# hyperparameters for the model

model = Model(projection_head, input_size_classifier, nb_classes).to(device)

criterion_ss = NTXentLoss()
optimizer_ss = torch.optim.SGD(list(model.backbone.parameters()) + list(model.projection_head.parameters()), 
                               config.optimizer_ss.params.lr, config.optimizer_ss.params.momentum, 
                               config.optimizer_ss.params.weight_decay)
scheduler_ss = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ss, T_max=nb_epochs_self_supervised_by_cycle*nb_cycles,
                                                          eta_min=config.scheduler_ss.params.eta_min, last_epoch=-1)

criterion_su = nn.CrossEntropyLoss()
optimizer_su = torch.optim.SGD(model.classifier.parameters(), config.optimizer_su.params.lr, 
                               config.optimizer_su.params.momentum, config.optimizer_su.params.weight_decay)
scheduler_su = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_su, T_max=nb_epochs_supervised_by_cycle*nb_cycles,
                                                          eta_min=config.scheduler_su.params.eta_min, last_epoch=-1)

# training

for cycle in range (nb_cycles) :
    for epochs in range(nb_epochs_self_supervised_by_cycle) :
        sum_loss_ss = self_supervised_training(device, model, train_dataloader_self_supervised, criterion_ss, optimizer_ss, scheduler_ss)
        wandb.log({"loss self-supervised": sum_loss_ss/nb_steps,
                   "learning rate self-supervised": scheduler_ss.get_last_lr()[0]
                })
    for epochs in range(nb_epochs_supervised_by_cycle) :
        sum_loss_su, accuracy = supervised_training(device, model, train_dataloader_supervised, criterion_su, optimizer_su, 
                                                              scheduler_su, softmax, ressemblance_matrix, nb_experiences_by_class, confusion_matrix)
        ressemblance_matrix = ressemblance_matrix / nb_experiences_by_class.reshape(-1, 1)
        wandb.log({"loss supervised": sum_loss_su/nb_steps,
               "accuracy supervised": accuracy/(batch_size*nb_steps),
               "learning rate supervised": scheduler_su.get_last_lr()[0]
                })
    if augmentation_adjustments:
        compute_new_augmentations(nb_classes, power_list, ressemblance_matrix, threshold)
        update_new_augmentations(self_supervised_augmentations, power_list)
        wandb.log({"power class 0 ": power_list[0],
               "power class 1 ": power_list[1],
               "power class 2 ": power_list[2],
               "power class 3 ": power_list[3],
               "power class 4 ": power_list[4],
               "power class 5 ": power_list[5],
               "power class 6 ": power_list[6],
               "power class 7 ": power_list[7],
               "power class 8 ": power_list[8], 
               "power class 9 ": power_list[9],
                })
    ressemblance_matrix.fill_(0)
    nb_experiences_by_class.fill_(0)

# test

test_accuracy = test_fct(device, model, test_dataloader, ressemblance_matrix, nb_experiences_by_class)
wandb.log({"test_accuracy": test_accuracy})
