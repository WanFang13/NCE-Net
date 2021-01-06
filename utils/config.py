''' Configuration File.
'''


# set random seed
def set_random_seed(seed):
    import torch
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DATASETS(object):
    def __init__(self, dataset='cifar10', num_classes=10):
        self.DATASET = dataset
        self.NUM_CLASS = num_classes
        self.NUM_TRAIN = 50000
        self.ROOT = {
            'cifar10': '../Data',
            'cifar100': '../Data'
        }


class ACTIVE_LEARNING(object):
    def __init__(self):
        self.TRIALS = 3
        self.CYCLES = 10
        self.ADDENDUM = 1000
        self.SUBSET = 10000

        self.NUM_PROTO = 1
        self.WEIGHT = 1.0
        self.ALPHA = 0.25


class TRAIN(object):
    def __init__(self):
        self.BATCH = 128
        self.EPOCH = 200
        self.LR = 0.1
        self.MILESTONES = [160]
        self.MOMENTUM = 0.9
        self.WDECAY = 5e-4


class global_vars(object):
    def __init__(self):
        self.iter = 0

    def update_vars(self):
        self.iter += 1

    def reset_vars(self):
        self.iter = 0


class CONFIG(object):
    def __init__(self, dataset='cifar10', num_classes=10):
        self.DATASET = DATASETS(dataset=dataset, num_classes=num_classes)
        self.ACTIVE_LEARNING = ACTIVE_LEARNING()
        self.TRAIN = TRAIN()
        self.global_iter = global_vars()


def get_configs(dataset='cifar10', num_classes=10):
    cfg = CONFIG(dataset=dataset, num_classes=num_classes)
    return cfg

