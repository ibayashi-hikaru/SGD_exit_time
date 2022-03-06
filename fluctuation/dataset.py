from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
class CIFAR2(Dataset):
    """Face Landmarks dataset."""

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
