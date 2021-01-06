
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as T
from torchvision.datasets import CIFAR100, CIFAR10
from models.sampler import SubsetSequentialSampler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


def get_transform(dataset):
    if dataset == 'cifar10':
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        return train_transform, test_transform
    if dataset == 'cifar100':
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        return train_transform, test_transform
    else:
        print("Error: No dataset named {}!".format(dataset))
        return -1


def get_dataset(dataset_root, dataset, train_transform, test_transform):
    if dataset == 'cifar10':
        train = CIFAR10(dataset_root, train=True, download=True, transform=train_transform)
        unlabeled = CIFAR10(dataset_root, train=True, download=True, transform=train_transform)
        test = CIFAR10(dataset_root, train=False, download=True, transform=test_transform)
    elif dataset == 'cifar100':
        train = CIFAR100(dataset_root, train=True, download=True, transform=train_transform)
        unlabeled = CIFAR100(dataset_root, train=True, download=True, transform=train_transform)
        test = CIFAR100(dataset_root, train=False, download=True, transform=test_transform)
    else:
        print("Error: No dataset named {}!".format(dataset))
        return -1
    return train, test, unlabeled


def get_training_functions(cfg, models):
    # criterion = nn.MSELoss()
    criterion = nn.BCELoss()
    optim_backbone = optim.SGD(models['backbone'].parameters(), lr=cfg.TRAIN.LR,
                               momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WDECAY)
    sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=cfg.TRAIN.MILESTONES)

    optimizers = {'backbone': optim_backbone}
    schedulers = {'backbone': sched_backbone}
    return criterion, optimizers, schedulers


def compute_confusion(score):

    max_score = score.max(1, keepdim=True)[0]
    confusion = (1 - max_score + score).sum(1)

    return confusion


def get_selection_metric(models, unlabeled_loader):
    models['backbone'].eval()

    with torch.no_grad():
        rejection = torch.tensor([]).cuda()
        # confusion = torch.tensor([]).cuda()
        for (inputs, label) in unlabeled_loader:
            inputs = inputs.cuda()
            score, _ = models['backbone'](inputs)

            rejection = torch.cat((rejection, (1-score).sum(1)), 0)
            # confusion = torch.cat((confusion, compute_confusion(score)), 0)

    return rejection.cpu()


def update_dataloaders(
        cfg,
        unlabeled_set, labeled_set,
        unlabeled_dataset, train_dataset,
        models, dataloaders
):
    # Randomly sample 10000 unlabeled data points
    random.shuffle(unlabeled_set)
    subset = unlabeled_set[:cfg.ACTIVE_LEARNING.SUBSET]

    # Create unlabeled dataloader for the unlabeled subset
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=cfg.TRAIN.BATCH,
                                  sampler=SubsetSequentialSampler(subset),
                                  pin_memory=True)

    # select
    metric = get_selection_metric(models, unlabeled_loader)
    arg = np.argsort(metric)

    # Update the labeled dataset and the unlabeled dataset, respectively
    budget = cfg.ACTIVE_LEARNING.ADDENDUM
    labeled_set += list(torch.tensor(subset)[arg][-budget:].numpy())
    unlabeled_set = list(torch.tensor(subset)[arg][:-budget].numpy()) + unlabeled_set[cfg.ACTIVE_LEARNING.SUBSET:]

    # Create a new dataloader for the updated labeled dataset
    dataloaders['train'] = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH,
                                      sampler=SubsetRandomSampler(labeled_set),
                                      pin_memory=True)

    return dataloaders, unlabeled_loader, unlabeled_set
