import argparse
import numpy as np

def get_args():
    dataset_list = [
        "uniform-cifar10",
        "non_uniform-cifar10-noiseless",
        "non_uniform-cifar10-noisy",
        "clcifar10",
        "clcifar10-aggregate",
        "clcifar10-noiseless",
        "clcifar10-iid",
        "clcifar20",
        "clcifar20-aggregate",
        "clcifar20-noiseless",
        "clcifar20-iid",
        "uniform-cifar20",
        "fwd-original-with0",
        "fwd-original-without0",
        "clcifar10-strong",
        "clcifar10-mcl",
        "clcifar20-mcl",
        "noisy-uniform-cifar10",
        "noisy-uniform-cifar20"
    ]

    algo_list = [
        "scl-exp",
        "scl-nl",
        "ure-ga-u",
        "ure-ga-r",
        "fwd-u",
        "fwd-r",
        "cpe-i",
        "l-w",
        "l-uw",
        "scl-nl-w",
        "pc-sigmoid"
    ]

    model_list = [
        "resnet18",
        "m-resnet18"
    ]

    parser = argparse.ArgumentParser()

    parser.add_argument('--algo', type=str, choices=algo_list, help='Algorithm')
    parser.add_argument('--dataset_name', type=str, choices=dataset_list, help='Dataset name')
    parser.add_argument('--model', type=str, choices=model_list, help='Model name')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--data_aug', type=str)
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--data_cleaning_rate', type=float, default=1)
    parser.add_argument('--num_cl', type=float, default=2)
    parser.add_argument('--valid_ratio', type=float, default=1)
    parser.add_argument('--eta', type=float, default=0)

    args = parser.parse_args()
    return args

def get_dataset_T(dataset, num_classes):
    dataset_T = np.zeros((num_classes,num_classes))
    class_count = np.zeros(num_classes)
    for i in range(len(dataset)):
        dataset_T[dataset.dataset.ord_labels[i]][dataset.dataset.targets[i]] += 1
        class_count[dataset.dataset.ord_labels[i]] += 1
    for i in range(num_classes):
        dataset_T[i] /= class_count[i]
    return dataset_T
