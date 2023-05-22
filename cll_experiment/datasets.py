import os
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
import urllib.request
import pickle

def get_dataset(args):
    dataset_name = args.dataset_name
    data_aug = args.data_aug
    data_cleaning_rate = args.data_cleaning_rate
    num_cl = args.num_cl
    eta = args.eta
    if dataset_name == "uniform-cifar10":
        trainset, validset, testset, ord_trainset, ord_validset = get_cifar10("uniform", None, None, data_aug)
    elif dataset_name == "uniform-cifar20":
        trainset, validset, testset, ord_trainset, ord_validset = get_cifar20(data_aug)
    elif dataset_name == "non_uniform-cifar10-noiseless":
        trainset, validset, testset, ord_trainset, ord_validset = get_cifar10("real", "random-1_T.npy", "zero", data_aug)
    elif dataset_name == "non_uniform-cifar10-noisy":
        trainset, validset, testset, ord_trainset, ord_validset = get_cifar10("real", "random-1_T.npy", None, data_aug)
    elif dataset_name == "clcifar10":
        trainset, validset, testset, ord_trainset, ord_validset = get_clcifar10(data_aug=data_aug, valid_ratio=args.valid_ratio)
    elif dataset_name == "clcifar10-aggregate":
        trainset, validset, testset, ord_trainset, ord_validset = get_clcifar10(data_aug, "aggregate")
    elif dataset_name == "clcifar10-noiseless":
        trainset, validset, testset, ord_trainset, ord_validset = get_clcifar10(data_aug, "noiseless", data_cleaning_rate=data_cleaning_rate)
    elif dataset_name == "clcifar10-iid":
        trainset, validset, testset, ord_trainset, ord_validset = get_clcifar10(data_aug, "iid")
    elif dataset_name == "clcifar20":
        trainset, validset, testset, ord_trainset, ord_validset = get_clcifar20(data_aug)
    elif dataset_name == "clcifar20-aggregate":
        trainset, validset, testset, ord_trainset, ord_validset = get_clcifar20(data_aug, "aggregate")
    elif dataset_name == "clcifar20-noiseless":
        trainset, validset, testset, ord_trainset, ord_validset = get_clcifar20(data_aug, "noiseless", data_cleaning_rate=data_cleaning_rate)
    elif dataset_name == "clcifar20-iid":
        trainset, validset, testset, ord_trainset, ord_validset = get_clcifar20(data_aug, "iid")
    elif dataset_name[:3] == "fwd":
        trainset, validset, testset, ord_trainset, ord_validset = get_cifar10(dataset_name, None, None, data_aug)
    elif dataset_name == "clcifar10-strong":
        trainset, validset, testset, ord_trainset, ord_validset = get_clcifar10(data_aug, "strong")
    elif dataset_name == "clcifar10-mcl":
        trainset, validset, testset, ord_trainset, ord_validset = get_clcifar10(data_aug, "mcl", data_cleaning_rate=data_cleaning_rate, num_cl=num_cl)
    elif dataset_name == "clcifar20-mcl":
        trainset, validset, testset, ord_trainset, ord_validset = get_clcifar20(data_aug, "mcl", data_cleaning_rate=data_cleaning_rate, num_cl=num_cl)
    elif dataset_name == "noisy-uniform-cifar10":
        trainset, validset, testset, ord_trainset, ord_validset = get_cifar10("synthetic-noise", None, None, data_aug, eta)
    else:
        raise NotImplementedError
    return trainset, validset, testset, ord_trainset, ord_validset

def get_cifar10(T_option, T_filepath=None, noise_level=None, data_aug=False, eta=0):
    if data_aug:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4922, 0.4832, 0.4486], [0.2456, 0.2419, 0.2605]
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4922, 0.4832, 0.4486], [0.2456, 0.2419, 0.2605]
                ),
            ]
        )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.4922, 0.4832, 0.4486], [0.2456, 0.2419, 0.2605]
            ),
        ]
    )
    
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    n_samples = len(dataset)
    
    ord_trainset, ord_validset = torch.utils.data.random_split(dataset, [int(n_samples*0.9), n_samples - int(n_samples*0.9)])
    
    trainset = deepcopy(ord_trainset)
    validset = deepcopy(ord_validset)
    trainset.dataset.ord_labels = deepcopy(trainset.dataset.targets)
    validset.dataset.ord_labels = deepcopy(validset.dataset.targets)
    num_classes = 10
    
    if T_option == "uniform":
        T = torch.full([num_classes, num_classes], 1/(num_classes-1))
        for i in range(num_classes):
            T[i][i] = 0
    elif T_option == "real":
        T = np.load(T_filepath)
    elif T_option == "fwd-original-with0":
        T = np.zeros([num_classes, num_classes])
        for i in range(num_classes):
            select_class = np.random.choice([k for k in range(10) if k != i], size=3, replace=False)
            probs = np.random.dirichlet(np.ones(3))
            for j in range(3):
                T[i][select_class[j]] = probs[j]
            T[i] /= sum(T[i])
    elif T_option == "fwd-original-without0":
        T = np.zeros([num_classes, num_classes])
        for i in range(num_classes):
            indexes = [k for k in range(10) if k != i]
            np.random.shuffle(indexes)
            for j in range(3):
                T[i][indexes[j*3]] = 0.6/3
                T[i][indexes[j*3+1]] = 0.3/3
                T[i][indexes[j*3+2]] = 0.1/3
            T[i] /= sum(T[i])
    elif T_option == "synthetic-noise":
        T = np.array(torch.full([num_classes, num_classes], (1-eta)/(num_classes-1)))
        for i in range(num_classes):
            T[i][i] = eta
        for i in range(num_classes):
            T[i] /= sum(T[i])
    else:
        raise NotImplementedError
    
    if noise_level == "zero":
        for i in range(10):
            T[i][i] = 0
            T[i] = T[i] / sum(T[i])
    
    for i in range(n_samples):
        ord_label = trainset.dataset.targets[i]
        trainset.dataset.targets[i] = np.random.choice(list(range(10)), p=T[ord_label])
    
    for i in range(n_samples):
        ord_label = validset.dataset.targets[i]
        validset.dataset.targets[i] = np.random.choice(list(range(10)), p=T[ord_label])
    
    return trainset, validset, testset, ord_trainset, ord_validset

def get_cifar20(data_aug=False):
    if data_aug:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5068, 0.4854, 0.4402], [0.2672, 0.2563, 0.2760]
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5068, 0.4854, 0.4402], [0.2672, 0.2563, 0.2760]
                ),
            ]
        )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5068, 0.4854, 0.4402], [0.2672, 0.2563, 0.2760]
            ),
        ]
    )
    
    dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    n_samples = len(dataset)
    num_classes = 20

    def _cifar100_to_cifar20(target):
        _dict = {0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3, 10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11, 20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15, 30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5, 40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10, 50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17, 60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19, 70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13, 80: 16, 81: 18, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19, 90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13}
        return _dict[target]
    
    dataset.targets = [_cifar100_to_cifar20(i) for i in dataset.targets]
    testset.targets = [_cifar100_to_cifar20(i) for i in testset.targets]
    ord_trainset, ord_validset = torch.utils.data.random_split(dataset, [int(n_samples*0.9), n_samples - int(n_samples*0.9)])
    
    trainset = deepcopy(ord_trainset)
    validset = deepcopy(ord_validset)
    trainset.dataset.ord_labels = deepcopy(trainset.dataset.targets)
    validset.dataset.ord_labels = deepcopy(validset.dataset.targets)
    
    T = torch.full([num_classes, num_classes], 1/(num_classes-1))
    for i in range(num_classes):
        T[i][i] = 0
    
    for i in range(n_samples):
        ord_label = trainset.dataset.targets[i]
        trainset.dataset.targets[i] = np.random.choice(list(range(20)), p=T[ord_label])
    
    for i in range(n_samples):
        ord_label = validset.dataset.targets[i]
        validset.dataset.targets[i] = np.random.choice(list(range(20)), p=T[ord_label])
    
    return trainset, validset, testset, ord_trainset, ord_validset

def choose_comp_label(labels, ord_label, noise_level):
    if noise_level == "random-1":
        return labels[0]
    elif noise_level == "random-2":
        return labels[1]
    elif noise_level == "random-3":
        return labels[2]
    elif noise_level == "aggregate":
        for cl in labels:
            if labels.count(cl) > 1:
                return cl
        return np.random.choice(labels)
    elif noise_level == "worst":
        if labels.count(ord_label) > 0:
            return ord_label
        return np.random.choice(labels)
    elif noise_level == "noiseless":
        if labels[0] != ord_label:
            return labels[0]
        else:
            return None
    elif noise_level == "multiple_cl":
        return labels
    else:
        print('noise_level', noise_level)
        raise NotImplementedError

class CustomDataset(Dataset):
    def __init__(self, root="./data", noise_level="random-1", transform=None, dataset_name="clcifar10", data_cleaning_rate=None, num_cl=None):

        os.makedirs(os.path.join(root, dataset_name), exist_ok=True)
        dataset_path = os.path.join(root, dataset_name, f"{dataset_name}.pkl")

        if not os.path.exists(dataset_path):
            if dataset_name == "clcifar10":
                print("Downloading clcifar10(148.3MB)")
                url = "https://clcifar.s3.us-west-2.amazonaws.com/clcifar10.pkl"
                with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=url.split('/')[-1]) as t:
                    urllib.request.urlretrieve(url, dataset_path, reporthook=lambda b, bsize, tsize: t.update(bsize))
            elif dataset_name == "clcifar20":
                print("Downloading clcifar20(150.6MB)")
                url = "https://clcifar.s3.us-west-2.amazonaws.com/clcifar20.pkl"
                with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=url.split('/')[-1]) as t:
                    urllib.request.urlretrieve(url, dataset_path, reporthook=lambda b, bsize, tsize: t.update(bsize))
            else:
                raise NotImplementedError

        data = pickle.load(open(dataset_path, "rb"))

        self.transform = transform
        self.input_dim = 3 * 32 * 32
        self.num_classes = 10 if dataset_name == "clcifar10" else 20

        self.targets = []
        self.data = []
        self.ord_labels = []
        print("CLCIFAR noise level:", noise_level)
        if noise_level == "noiseless":
            print("Data cleaning rate:", data_cleaning_rate)
        # if noise_level == "strong":
        #     selected = [False] * 50000
        #     indexes = list(range(10))
        #     np.random.shuffle(indexes)

        #     for k in indexes:
        #         for i in range(50000):
        #             if not selected[i] and k != data['ord_labels'] and k in data['cl_labels'][i]:
        #                 selected[i] = True
        #                 self.targets.append(k)
        #                 self.data.append(data["images"][i])
        #                 self.ord_labels.append(data["ord_labels"][i])
        if noise_level == "iid":
            targets = []
            true_labels = []
            for i in range(len(data["cl_labels"])):
                cl = choose_comp_label(data["cl_labels"][i], data["ord_labels"][i], "aggregate")
                if cl is not None:
                    targets.append(cl)
                    true_labels.append(data["ord_labels"][i])
            T = np.zeros([self.num_classes, self.num_classes])
            for i in range(len(targets)):
                T[true_labels[i]][targets[i]] += 1
            for i in range(self.num_classes):
                T[i] /= sum(T[i])
            for i in range(len(data["cl_labels"])):
                self.targets.append(np.random.choice(list(range(self.num_classes)), p=T[data["ord_labels"][i]]))
                self.data.append(data["images"][i])
                self.ord_labels.append(data["ord_labels"][i])
        elif noise_level == "mcl":
            for i in range(len(data["cl_labels"])):
                for j in range(num_cl):
                    self.targets.append(data["cl_labels"][i][j])
                    self.data.append(data["images"][i])
                    self.ord_labels.append(data["ord_labels"][i])
        else:
            noise = {'targets':[], 'data':[], 'ord_labels':[]}
            for i in range(len(data["cl_labels"])):
                cl = choose_comp_label(data["cl_labels"][i], data["ord_labels"][i], noise_level)
                if cl is not None:
                    self.targets.append(cl)
                    self.data.append(data["images"][i])
                    self.ord_labels.append(data["ord_labels"][i])
                else:
                    if noise_level == "noiseless":
                        noise['targets'].append(data["cl_labels"][i][0])
                        noise['data'].append(data["images"][i])
                        noise['ord_labels'].append(data["ord_labels"][i])
        if noise_level == "noiseless":
            assert((0 <= data_cleaning_rate) and (data_cleaning_rate <= 1))
            noise_num = int(len(noise['data']) * (1-data_cleaning_rate))
            print(f"number of noise added: {noise_num}/{len(noise['data'])}")
            self.targets.extend(noise['targets'][:noise_num])
            self.data.extend(noise['data'][:noise_num])
            self.ord_labels.extend(noise['ord_labels'][:noise_num])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]

# def get_clcifar10_trainset(root="./data", noise_level="random-1", transform=None):
#     return CustomDataset(root=root, noise_level=noise_level, transform=transform)

def get_clcifar10(data_aug=False, noise_level="random-1", data_cleaning_rate=None, num_cl=None, valid_ratio=1.0):
    if data_aug:
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                ),
            ]
        )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
            ),
        ]
    )
    
    # if bias == "strong":
    #     dataset = get_clcifar10_trainset(root='./data', noise_level="strong", transform=transform)
    dataset = CustomDataset(root='./data', noise_level=noise_level, transform=transform, dataset_name="clcifar10", data_cleaning_rate=data_cleaning_rate, num_cl=num_cl)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    n_samples = len(dataset)
    trainset_size = 45000
    validset_size = int(5000 * valid_ratio)
    
    ord_trainset, ord_validset, _ = torch.utils.data.random_split(dataset, [trainset_size, validset_size, n_samples - (trainset_size + validset_size)])
    
    trainset = deepcopy(ord_trainset)
    validset = deepcopy(ord_validset)

    ord_trainset.dataset.targets = ord_trainset.dataset.ord_labels
    ord_validset.dataset.targets = ord_validset.dataset.ord_labels
    
    return trainset, validset, testset, ord_trainset, ord_validset

def get_clcifar20(data_aug=False, noise_level="random-1", bias=None, data_cleaning_rate=None, num_cl=None):
    if data_aug:
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5068, 0.4854, 0.4402], [0.2672, 0.2563, 0.2760]
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5068, 0.4854, 0.4402], [0.2672, 0.2563, 0.2760]
                ),
            ]
        )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5068, 0.4854, 0.4402], [0.2672, 0.2563, 0.2760]
            ),
        ]
    )
    num_classes = 20
    
    dataset = CustomDataset(root='./data', noise_level=noise_level, transform=transform, dataset_name="clcifar20", data_cleaning_rate=data_cleaning_rate, num_cl=num_cl)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    n_samples = len(dataset)

    def _cifar100_to_cifar20(target):
        _dict = {0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3, 10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11, 20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15, 30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5, 40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10, 50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17, 60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19, 70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13, 80: 16, 81: 18, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19, 90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13}
        return _dict[target]
    
    testset.targets = [_cifar100_to_cifar20(i) for i in testset.targets]
    ord_trainset, ord_validset = torch.utils.data.random_split(dataset, [int(n_samples*0.9), n_samples - int(n_samples*0.9)])
    
    trainset = deepcopy(ord_trainset)
    validset = deepcopy(ord_validset)
    
    ord_trainset.dataset.targets = ord_trainset.dataset.ord_labels
    ord_validset.dataset.targets = ord_validset.dataset.ord_labels
    
    return trainset, validset, testset, ord_trainset, ord_validset