
import logging
import torch
import torch.nn.functional as F


def print_class_object(obj, name, logger):
    for key, value in obj.__dict__.items():
        logger.info("CONFIG -- {} - {}: {}".format(name, key, value))


def print_config(cfg, logger):
    print_class_object(cfg.DATASET, 'DATASET', logger)
    print_class_object(cfg.ACTIVE_LEARNING, 'ACTIVE_LEARNING', logger)
    print_class_object(cfg.TRAIN, 'TRAIN', logger)


class Logger:
    def __init__(self, method='ActiveLearning', logname='./output/log.txt'):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(logname)
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s-["+method+"]: %(message)s", "%Y%m%d-%H:%M:%S"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    @property
    def get_log(self):
        return self.logger


def get_one_hot_label(labels=None, num_classes=10):

    return torch.zeros(labels.shape[0],
                       num_classes
                       ).cuda().scatter_(1, labels.view(-1, 1), 1)


def normed_euclidean_distance_func(feat1, feat2):
    # Normalized Euclidean Distance
    # feat1: N * Dim
    # feat2: M * Dim
    # out:   N * M Euclidean Distance
    feat1, feat2 = F.normalize(feat1), F.normalize(feat2)
    feat_matmul  = torch.matmul(feat1, feat2.t())
    distance = torch.ones_like(feat_matmul) - feat_matmul
    distance = distance * 2
    return distance.clamp(1e-10).sqrt()


def euclidean_distance_func(feat1, feat2):
    # Euclidean Distance
    # feat1: N * Dim
    # feat2: M * Dim
    # out:   N * M Euclidean Distance
    feat1_square = torch.sum(torch.pow(feat1, 2), 1, keepdim=True)
    feat2_square = torch.sum(torch.pow(feat2, 2), 1, keepdim=True)
    feat_matmul  = torch.matmul(feat1, feat2.t())
    distance = feat1_square + feat2_square.t() - 2 * feat_matmul
    return distance.clamp(1e-10).sqrt()


def cosine_distance_func(feat1, feat2):
    # feat1: N * Dim
    # feat2: M * Dim
    # out:   N * M Cosine Distance
    distance = torch.matmul(F.normalize(feat1), F.normalize(feat2).t())
    return distance


def cosine_distance_full_func(feat1, feat2):
    # feat1: N * Dim
    # feat2: M * Dim
    # out:   (N+M) * (N+M) Cosine Distance
    feat = torch.cat((feat1, feat2), dim=0)
    distance = torch.matmul(F.normalize(feat), F.normalize(feat).t())
    return distance

