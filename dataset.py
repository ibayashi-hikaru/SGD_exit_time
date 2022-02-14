from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
class CIFAR2(Dataset):
    def __init__(self, root, train, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform
        if train:
            self.data = torch.load(root+"/CIFAR2/train_data.pt")
            self.targets = torch.load(root+"/CIFAR2/train_targets.pt")
        else:
            self.data = torch.load(root+"/CIFAR2/test_data.pt")
            self.targets = torch.load(root+"/CIFAR2/test_targets.pt")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

import pandas as pd
import numpy as np
import torch.nn.functional as F
class AVILA2(Dataset):

    def __init__(self, root, train, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform
        names = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','class']
        if train:
            train_data = pd.read_csv("DOWNLOADs/AVILA/avila-tr.txt",header=None,names=names)
            X_train = train_data.iloc [:, [0,1,2,3,4,5,6,7,8,9]] 
            Y_train = train_data.iloc [:, [10]]
            #
            self.data = torch.tensor(X_train.values).type(torch.FloatTensor)
            for i in range(self.data.shape[0]):
                self.data[i,:] = self.data[i,:] - torch.mean(self.data[i,:])
            self.data = torch.nn.functional.normalize(self.data)
            #
            binary_label = list(map(lambda x: (ord(x) - 65)%2, Y_train.values[:,0]))
            self.targets = F.one_hot(torch.tensor(binary_label))
        else:
            test_data  = pd.read_csv("DOWNLOADs/AVILA/avila-ts.txt")
            X_test = test_data.iloc[:, [0,1,2,3,4,5,6,7,8,9]] 
            Y_test = test_data.iloc[:, [10]]
            #
            self.data = torch.tensor(X_test.values).type(torch.FloatTensor)
            for i in range(self.data.shape[0]):
                self.data[i,:] = self.data[i,:] - torch.mean(self.data[i,:])
            self.data = torch.nn.functional.normalize(self.data)
            self.data = torch.tensor(X_test.values).type(torch.FloatTensor)
            #
            binary_label = list(map(lambda x: (ord(x) - 65)%2, Y_test.values[:,0]))
            self.targets = F.one_hot(torch.tensor(binary_label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target
