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
import wandb
import argparse

num_classes = 10
eval_n_epoch = 5
epochs = 300
batch_size = 512
num_workers = 4
device = "cuda"
loss_func = nn.CrossEntropyLoss(reduction='none')

def get_cifar10(T_option, T_filepath, noise_level, data_aug=False):
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
    n_samples = 50000
    
    ord_trainset, ord_validset = torch.utils.data.random_split(dataset, [int(n_samples*0.9), int(n_samples*0.1)])
    
    trainset = deepcopy(ord_trainset)
    validset = deepcopy(ord_validset)
    trainset.dataset.ord_labels = deepcopy(trainset.dataset.targets)
    validset.dataset.ord_labels = deepcopy(validset.dataset.targets)
    
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
    n_samples = 50000
    global num_classes
    num_classes = 20

    def _cifar100_to_cifar20(target):
        _dict = {0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3, 10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11, 20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15, 30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5, 40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10, 50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17, 60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19, 70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13, 80: 16, 81: 18, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19, 90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13}
        return _dict[target]
    
    dataset.targets = [_cifar100_to_cifar20(i) for i in dataset.targets]
    testset.targets = [_cifar100_to_cifar20(i) for i in testset.targets]
    ord_trainset, ord_validset = torch.utils.data.random_split(dataset, [int(n_samples*0.9), int(n_samples*0.1)])
    
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
        correct_cl = []
        for cl in labels:
            if cl != ord_label:
                correct_cl.append(cl)
        if len(correct_cl) == 0:
            return [i for i in range(num_classes) if i != ord_label][np.random.randint(0, num_classes)]
        return np.random.choice(correct_cl)
    elif noise_level == "multiple_cl":
        return labels
    else:
        raise NotImplementedError

class CustomDataset(Dataset):
    def __init__(self, root="./data", noise_level="random-1", transform=None, dataset_name="clcifar10"):

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
            T = np.zeros([num_classes, num_classes])
            for i in range(len(targets)):
                T[true_labels[i]][targets[i]] += 1
            for i in range(num_classes):
                T[i] /= sum(T[i])
            for i in range(len(data["cl_labels"])):
                self.targets.append(np.random.choice(list(range(num_classes)), p=T[data["ord_labels"][i]]))
                self.data.append(data["images"][i])
                self.ord_labels.append(data["ord_labels"][i])
        else:
            for i in range(len(data["cl_labels"])):
                cl = choose_comp_label(data["cl_labels"][i], data["ord_labels"][i], noise_level)
                if cl is not None:
                    self.targets.append(cl)
                    self.data.append(data["images"][i])
                    self.ord_labels.append(data["ord_labels"][i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]

# def get_clcifar10_trainset(root="./data", noise_level="random-1", transform=None):
#     return CustomDataset(root=root, noise_level=noise_level, transform=transform)

def get_clcifar10(data_aug=False, noise_level="random-1", bias=None):
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
    dataset = CustomDataset(root='./data', noise_level=noise_level, transform=transform, dataset_name="clcifar10")
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    n_samples = 50000
    
    ord_trainset, ord_validset = torch.utils.data.random_split(dataset, [int(n_samples*0.9), int(n_samples*0.1)])
    
    trainset = deepcopy(ord_trainset)
    validset = deepcopy(ord_validset)
    
    ord_trainset.dataset.targets = ord_trainset.dataset.ord_labels
    ord_validset.dataset.targets = ord_validset.dataset.ord_labels
    
    return trainset, validset, testset, ord_trainset, ord_validset

def get_clcifar20(data_aug=False, noise_level="random-1", bias=None):
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
    global num_classes
    num_classes = 20
    
    dataset = CustomDataset(root='./data', noise_level=noise_level, transform=transform, dataset_name="clcifar20")
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    n_samples = 50000

    def _cifar100_to_cifar20(target):
        _dict = {0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3, 10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11, 20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15, 30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5, 40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10, 50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17, 60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19, 70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13, 80: 16, 81: 18, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19, 90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13}
        return _dict[target]
    
    testset.targets = [_cifar100_to_cifar20(i) for i in testset.targets]
    ord_trainset, ord_validset = torch.utils.data.random_split(dataset, [int(n_samples*0.9), int(n_samples*0.1)])
    
    trainset = deepcopy(ord_trainset)
    validset = deepcopy(ord_validset)
    
    ord_trainset.dataset.targets = ord_trainset.dataset.ord_labels
    ord_validset.dataset.targets = ord_validset.dataset.ord_labels
    
    return trainset, validset, testset, ord_trainset, ord_validset

def get_resnet18():
    resnet = torchvision.models.resnet18(weights=None)
    # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # resnet.maxpool = nn.Identity()
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)
    return resnet

def get_modified_resnet18():
    resnet = torchvision.models.resnet18(weights=None)
    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet.maxpool = nn.Identity()
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)
    return resnet

def validate(model, dataloader):
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total

def compute_ure(outputs, labels, dataset_T):
    with torch.no_grad():
        outputs = -F.log_softmax(outputs, dim=1)
        if torch.det(dataset_T) != 0:
            T_inv = torch.inverse(dataset_T).to(device)
        else:
            T_inv = torch.pinverse(dataset_T).to(device)
        loss_mat = torch.mm(outputs, T_inv.transpose(1, 0))
        ure = -F.nll_loss(loss_mat, labels)
        return ure

def ga_loss(outputs, labels, class_prior, T):
    if torch.det(T) != 0:
        Tinv = torch.inverse(T)
    else:
        Tinv = torch.pinverse(T)
    batch_size = outputs.shape[0]
    outputs = -F.log_softmax(outputs, dim=1)
    loss_mat = torch.zeros([num_classes, num_classes], device=device)
    for k in range(num_classes):
        mask = k == labels
        indexes = torch.arange(batch_size).to(device)
        indexes = torch.masked_select(indexes, mask)
        if indexes.shape[0] > 0:
            outputs_k = outputs[indexes]
            # outputs_k = torch.gather(outputs, 0, indexes.view(-1, 1).repeat(1,num_classes))
            loss_mat[k] = class_prior[k] * outputs_k.mean(0)
    loss_vec = torch.zeros(num_classes, device=device)
    for k in range(num_classes):
        loss_vec[k] = torch.inner(Tinv[k], loss_mat[k])
    return loss_vec

def get_dataset_T(dataset):
    dataset_T = np.zeros((num_classes,num_classes))
    class_count = np.zeros(num_classes)
    for i in range(len(dataset)):
        dataset_T[dataset.dataset.ord_labels[i]][dataset.dataset.targets[i]] += 1
        class_count[dataset.dataset.ord_labels[i]] += 1
    for i in range(num_classes):
        dataset_T[i] /= class_count[i]
    return dataset_T

def compute_scel(outputs, labels, algo, dataset_T):
    outputs = outputs.softmax(dim=1)
    if algo[:3] != "cpe":
        outputs = torch.mm(outputs, dataset_T)
    outputs = (outputs + 1e-6).log()
    return F.nll_loss(outputs, labels)

def cpe_decode(model, dataloader):
    total = 0
    correct = 0
    U = torch.ones((num_classes, num_classes), device=device) * 1/9
    for i in range(num_classes):
        U[i][i] = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.view(-1, num_classes, 1).repeat(1, 1, num_classes) - U.expand(outputs.shape[0], num_classes, num_classes)
            predicted = torch.argmin(outputs.norm(dim=1), dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total

def train(args):
    dataset_name = args.dataset_name
    algo = args.algo
    model = args.model
    lr = args.lr
    seed = args.seed
    data_aug = True if args.data_aug.lower()=="true" else False

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if data_aug:
        print("Use data augmentation.")

    if dataset_name == "uniform-cifar10":
        trainset, validset, testset, ord_trainset, ord_validset = get_cifar10("uniform", None, None, data_aug)
    elif dataset_name == "uniform-cifar20":
        trainset, validset, testset, ord_trainset, ord_validset = get_cifar20(data_aug)
    elif dataset_name == "non_uniform-cifar10-noiseless":
        trainset, validset, testset, ord_trainset, ord_validset = get_cifar10("real", "random-1_T.npy", "zero", data_aug)
    elif dataset_name == "non_uniform-cifar10-noisy":
        trainset, validset, testset, ord_trainset, ord_validset = get_cifar10("real", "random-1_T.npy", None, data_aug)
    elif dataset_name == "clcifar10":
        trainset, validset, testset, ord_trainset, ord_validset = get_clcifar10(data_aug)
    elif dataset_name == "clcifar10-aggregate":
        trainset, validset, testset, ord_trainset, ord_validset = get_clcifar10(data_aug, "aggregate")
    elif dataset_name == "clcifar10-noiseless":
        trainset, validset, testset, ord_trainset, ord_validset = get_clcifar10(data_aug, "noiseless")
    elif dataset_name == "clcifar10-iid":
        trainset, validset, testset, ord_trainset, ord_validset = get_clcifar10(data_aug, "iid")
    elif dataset_name == "clcifar20":
        trainset, validset, testset, ord_trainset, ord_validset = get_clcifar20(data_aug)
    elif dataset_name == "clcifar20-aggregate":
        trainset, validset, testset, ord_trainset, ord_validset = get_clcifar20(data_aug, "aggregate")
    elif dataset_name == "clcifar20-noiseless":
        trainset, validset, testset, ord_trainset, ord_validset = get_clcifar20(data_aug, "noiseless")
    elif dataset_name == "clcifar20-iid":
        trainset, validset, testset, ord_trainset, ord_validset = get_clcifar20(data_aug, "iid")
    elif dataset_name[:3] == "fwd":
        trainset, validset, testset, ord_trainset, ord_validset = get_cifar10(dataset_name, None, None, data_aug)
    elif dataset_name == "clcifar10-strong":
        trainset, validset, testset, ord_trainset, ord_validset = get_clcifar10(data_aug, "strong")
    else:
        raise NotImplementedError

    # Print the complementary label distribution T
    dataset_T = get_dataset_T(trainset)
    np.set_printoptions(floatmode='fixed', precision=2)
    print("Dataset's transition matrix T:")
    print(dataset_T)
    dataset_T = torch.tensor(dataset_T, dtype=torch.float).to(device)

    # Set Q for forward algorithm
    if algo in ["fwd-u", "ure-ga-u"]:
        Q = torch.full([num_classes, num_classes], 1/(num_classes-1), device=device)
        for i in range(num_classes):
            Q[i][i] = 0
    elif algo in ["fwd-r", "ure-ga-r"]:
        Q = dataset_T
    if algo[:3] in ["fwd", "ure"]:
        print("The transition matrix Q for training:")
        print(np.array(Q.cpu()))

    # Print dataset's error rate
    count_wrong_label = np.zeros(num_classes)
    for i in range(len(trainset)):
        if trainset.dataset.targets[i] == trainset.dataset.ord_labels[i]:
            count_wrong_label[trainset.dataset.targets[i]] += 1
    print("Class error rate:")
    print(count_wrong_label / 4500)

    print(num_classes)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    ord_trainloader = DataLoader(ord_trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    ord_validloader = DataLoader(ord_validset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_labels = torch.tensor(np.array(trainset.dataset.targets), dtype=torch.int).squeeze()
    class_prior = train_labels.bincount().float() / train_labels.shape[0]

    if args.model == "resnet18":
        model = get_resnet18().to(device)
    elif args.model == "m-resnet18":
        model = get_modified_resnet18().to(device)
    else:
        raise NotImplementedError
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    validation_obj = ["valid_acc", "ure", "scel"]
    best_epoch = {obj: None for obj in validation_obj}

    wandb.login()
    if args.test:
        wandb.init(project="test", config={"lr": lr, "seed": seed}, tags=[algo])
    else:
        wandb.init(project=dataset_name, config={"lr": lr, "seed": seed}, tags=[algo])
    
    with tqdm(range(epochs), unit="epoch") as tepoch:
        # tepoch.set_description(f"lr={lr}")
        for epoch in tepoch:
            training_loss = 0.0
            scel = 0
            ure = 0
            model.train()

            for inputs, labels in trainloader:

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                ure += compute_ure(outputs, labels, dataset_T)

                scel += compute_scel(outputs, labels, algo, dataset_T)
                    
                if algo == "scl-exp":
                    outputs = F.softmax(outputs, dim=1)
                    loss = -F.nll_loss(outputs.exp(), labels)
                    loss.backward()
                
                elif algo[:6] == "ure-ga":
                    loss = ga_loss(outputs, labels, class_prior, Q)
                    if torch.min(loss) > 0:
                        loss = loss.sum()
                        loss.backward()
                    else:
                        beta_vec = torch.zeros(num_classes, requires_grad=True).to(device)
                        loss = torch.minimum(beta_vec, loss).sum() * -1
                        loss.backward()
                
                elif algo[:3] == "fwd":
                    q = torch.mm(F.softmax(outputs, dim=1), Q) + 1e-6
                    loss = F.nll_loss(q.log(), labels.squeeze())
                    loss.backward()
                
                elif algo == "cpe-i":
                    loss = F.cross_entropy(outputs, labels)
                    loss.backward()
                
                elif algo == "l-w":
                    outputs1 = 1 - F.softmax(outputs, dim=1)
                    loss = F.cross_entropy(outputs1, labels.squeeze(), reduction='none')
                    # p = (1 - F.softmax(outputs, dim=1) + 1e-6).log()
                    # loss = F.nll_loss(p, labels, reduction="none")
                    w = (1-F.softmax(outputs, dim=1)) / (num_classes-1)
                    w = 1-F.nll_loss(w, labels.squeeze(), reduction='none')
                    loss = (loss * w).mean()
                    loss.backward()

                elif algo == "scl-nl-w":
                    p = (1 - F.softmax(outputs, dim=1) + 1e-6).log()
                    loss = F.nll_loss(p, labels, reduction="none")
                    w = (1-F.softmax(outputs, dim=1)) / (num_classes-1)
                    w = 1-F.nll_loss(w, labels.squeeze(), reduction='none')
                    loss = (loss * w).mean()
                    loss.backward()
                
                elif algo == "l-uw":
                    outputs = 1 - F.softmax(outputs, dim=1)
                    loss = F.cross_entropy(outputs, labels.squeeze())
                    loss.backward()
                
                elif algo == "scl-nl":
                    p = (1 - F.softmax(outputs, dim=1) + 1e-6).log()
                    loss = F.nll_loss(p, labels)
                    loss.backward()

                else:
                    raise NotImplementedError
                
                optimizer.step()
                training_loss += loss.item()

                tepoch.set_postfix(loss=loss.item())

            ure /= len(trainloader)
            training_loss /= len(trainloader)
            scel /= len(trainloader)
            wandb.log({"ure": ure, "training_loss": training_loss, "scel": scel})

            if (epoch+1) % eval_n_epoch == 0:
                model.eval()
                if algo[:3] == "cpe":
                    train_acc, valid_acc, test_acc = cpe_decode(model, ord_trainloader), cpe_decode(model, ord_validloader), cpe_decode(model, testloader)
                else:
                    train_acc, valid_acc, test_acc = validate(model, ord_trainloader), validate(model, ord_validloader), validate(model, testloader)
                    print(train_acc, valid_acc, test_acc)
                wandb.log({"train_acc": train_acc, "valid_acc": valid_acc, "test_acc": test_acc})

                epoch_info = {
                    "epoch": epoch,
                    "train_acc": train_acc,
                    "valid_acc": valid_acc,
                    "test_acc": test_acc,
                    "ure": ure,
                    "scel": scel,
                    "training_loss": training_loss
                }
                if best_epoch["valid_acc"] is None or valid_acc > best_epoch["valid_acc"]["valid_acc"]:
                    best_epoch["valid_acc"] = epoch_info
                if best_epoch["ure"] is None or ure < best_epoch["ure"]["ure"]:
                    best_epoch["ure"] = epoch_info
                if best_epoch["scel"] is None or scel < best_epoch["scel"]["scel"]:
                    best_epoch["scel"] = epoch_info
    wandb.summary["best_epoch-valid_acc"] = best_epoch["valid_acc"]
    wandb.summary["best_epoch-ure"] = best_epoch["ure"]
    wandb.summary["best_epoch-scel"] = best_epoch["scel"]
    wandb.finish()  

if __name__ == "__main__":

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
        "clcifar10-strong"
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
        "scl-nl-w"
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

    args = parser.parse_args()

    train(args)
