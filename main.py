
import models.resnet_nce as nce
from utils.config import *
from utils.dataset import *
from utils.train_net import *
from utils.test_net import *
from utils.util import *


if __name__ == '__main__':

    # hyper params
    dataset = 'cifar10'
    num_classes = 10
    method = 'NCE-Net-Rejection'
    # method = 'temp'
    cfg = get_configs(dataset=dataset, num_classes=num_classes)
    time_str = time.strftime('-%Y%m%d-%H:%M:%S', time.localtime())

    # path and logger
    dataset_root = cfg.DATASET.ROOT[dataset]
    output_dir = 'output/' + dataset
    checkpoint_dir = os.path.join(output_dir, 'train', method)
    # checkpoint_dir = os.path.join(output_dir, 'train', method+time_str)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    log_save_file = checkpoint_dir + "/log.txt"
    my_logger = Logger(method, log_save_file).get_log
    my_logger.info("Checkpoint Path: {}".format(checkpoint_dir))
    print_config(cfg, my_logger)

    # define dataset and dataloaders
    train_transform, test_transform = get_transform(dataset)
    train_dataset, test_dataset, unlabeled_dataset = get_dataset(
        dataset_root, dataset, train_transform, test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH)

    # begin training
    Performance = np.zeros((cfg.ACTIVE_LEARNING.TRIALS, cfg.ACTIVE_LEARNING.CYCLES))
    for trial in range(cfg.ACTIVE_LEARNING.TRIALS):
        # initial
        set_random_seed(trial)
        indices = list(range(cfg.DATASET.NUM_TRAIN))
        random.shuffle(indices)
        labeled_set = indices[:cfg.ACTIVE_LEARNING.ADDENDUM]
        unlabeled_set = indices[cfg.ACTIVE_LEARNING.ADDENDUM:]

        train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH,
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True)
        dataloaders = {'train': train_loader, 'test': test_loader}

        # Model
        nce_net = nce.ResNet18(
            num_classes=num_classes,
            num_proto=cfg.ACTIVE_LEARNING.NUM_PROTO,
            alpha=cfg.ACTIVE_LEARNING.ALPHA
        ).cuda()
        models = {'backbone': nce_net}
        torch.backends.cudnn.benchmark = True

        # Active learning cycles
        for cycle in range(cfg.ACTIVE_LEARNING.CYCLES):
            criterion, optimizers, schedulers = get_training_functions(
                cfg, models
            )

            # Training and test
            train(
                cfg, my_logger, models, criterion, optimizers,
                schedulers, dataloaders,
                cfg.TRAIN.EPOCH, checkpoint_dir,
                trial, cycle
            )
            acc = test(models, dataloaders, mode='test')
            my_logger.info('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(
                trial + 1, cfg.ACTIVE_LEARNING.TRIALS, cycle + 1,
                cfg.ACTIVE_LEARNING.CYCLES, len(labeled_set), acc)
            )
            Performance[trial, cycle] = acc

            # update dataloaders
            dataloaders, unlabeled_loader, unlabeled_set = update_dataloaders(
                cfg,
                unlabeled_set, labeled_set,
                unlabeled_dataset, train_dataset,
                models, dataloaders
            )

        # Save a checkpoint
        torch.save(
            {
                'trial': trial + 1,
                'state_dict_backbone': models['backbone'].state_dict()
            },
            '{}/active_resnet18_cifar10_trial{}.pth'.format(checkpoint_dir, trial)
        )

