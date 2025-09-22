from PIL import Image
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment
from util.sampling import iid_sampling, non_iid_dirichlet_sampling, non_iid_dirichlet_sampling_returnphi, partition_test_data
import torch.utils

class Clothing(torch.utils.data.Dataset):
    def __init__(self, root, transform, mode):
        self.root = root
        self.noisy_labels = {}
        self.clean_labels = {}
        self.data = []
        self.targets = []
        self.transform = transform
        self.mode = mode

        with open(self.root + 'noisy_label_kv.txt', 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()
            img_path = self.root + entry[0]
            self.noisy_labels[img_path] = int(entry[1])

        with open(self.root + 'clean_label_kv.txt', 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()
            img_path = self.root + entry[0]
            self.clean_labels[img_path] = int(entry[1])

        if self.mode == 'train':
            with open(self.root + 'noisy_train_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            for l in lines:
                img_path = self.root + l
                self.data.append(img_path)
                target = self.noisy_labels[img_path]
                self.targets.append(target)
        elif self.mode == 'minitrain':
            with open(self.root + 'noisy_train_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            n = len(lines)
            np.random.seed(13)
            subset_idx = np.random.choice(n, int(n/10), replace=False)
            for i in subset_idx:
                l = lines[i]
                img_path = self.root + l
                self.data.append(img_path)
                target = self.noisy_labels[img_path]
                self.targets.append(target)
        elif self.mode == 'test':
            with open(self.root + 'clean_test_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            for l in lines:
                img_path = self.root + l
                self.data.append(img_path)
                target = self.clean_labels[img_path]
                self.targets.append(target)

    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.targets[index]
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.data)


def get_dataset(args):
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'cifar10':
        data_path = '../data/cifar10'
        args.num_classes = 10
        trans_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.2435, 0.2616)),
        ]) 

        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.2435, 0.2616))],
        )
        dataset_train = datasets.CIFAR10(data_path, train=True, download=False, transform=trans_train)
        dataset_train_noaug = datasets.CIFAR10(data_path, train=True, download=False, transform=trans_val)
        dataset_test = datasets.CIFAR10(data_path, train=False, download=False, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)
    elif args.dataset == 'cifar100':
        data_path = '../data/cifar100'
        args.num_classes = 100
        #args.model = 'resnet34'
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
        ])
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
        ])
        dataset_train = datasets.CIFAR100(data_path, train=True, download=False, transform=trans_train)
        dataset_train_noaug = datasets.CIFAR100(data_path, train=True, download=False, transform=trans_val)
        dataset_test = datasets.CIFAR100(data_path, train=False, download=False, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)

    elif args.dataset == 'clothing1m':
        data_path = '../clothing1M/'
        args.num_classes = 14
        #args.model = 'resnet50'
        trans_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        trans_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset_train = Clothing(data_path, trans_train, "train")
        dataset_train_noaug = Clothing(data_path, trans_val, "train")
        dataset_test = Clothing(data_path, trans_val, "test")
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)

    else:
        exit('Error: unrecognized dataset')


    if args.iid:
        dict_users = iid_sampling(n_train, args.num_users, args.seed)
    else:
        dict_users = non_iid_dirichlet_sampling(y_train, args.num_classes, args.non_iid_prob_class, args.num_users, args.seed, args.alpha_dirichlet)

    return dataset_train, dataset_train_noaug, dataset_test, dict_users

def get_dataset_localtest(args):
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'cifar10':
        data_path = '../data/cifar10'
        args.num_classes = 10
        trans_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.2435, 0.2616)),
        ]) 

        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.2435, 0.2616))],
        )
        dataset_train = datasets.CIFAR10(data_path, train=True, download=False, transform=trans_train)
        dataset_train_noaug = datasets.CIFAR10(data_path, train=True, download=False, transform=trans_val)
        dataset_test = datasets.CIFAR10(data_path, train=False, download=False, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)
    elif args.dataset == 'cifar100':
        data_path = '../data/cifar100'
        args.num_classes = 100
        #args.model = 'resnet34'
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
        ])
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
        ])
        dataset_train = datasets.CIFAR100(data_path, train=True, download=False, transform=trans_train)
        dataset_train_noaug = datasets.CIFAR100(data_path, train=True, download=False, transform=trans_val)
        dataset_test = datasets.CIFAR100(data_path, train=False, download=False, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)

    elif args.dataset == 'clothing1m':
        data_path = '../clothing1M/'
        args.num_classes = 14
        #args.model = 'resnet50'
        trans_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        trans_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset_train = Clothing(data_path, trans_train, "train")
        dataset_train_noaug = Clothing(data_path, trans_val, "train")
        dataset_test = Clothing(data_path, trans_val, "test")
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)

    else:
        exit('Error: unrecognized dataset')


    if args.iid:
        dict_users = iid_sampling(n_train, args.num_users, args.seed)
    else:
        dict_users, Phi = non_iid_dirichlet_sampling_returnphi(y_train, args.num_classes, args.non_iid_prob_class, args.num_users, args.seed, args.alpha_dirichlet)
    y_test = np.array(dataset_test.targets)
    dict_users_test = partition_test_data(y_test, y_train, dict_users, Phi, args.seed)
    return dataset_train, dataset_train_noaug, dataset_test, dict_users, dict_users_test




