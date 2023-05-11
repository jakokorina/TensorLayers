import torchvision.datasets as datasets
import torchvision.transforms as T

from torch.utils.data import DataLoader


def get_cifar10_dataloader(batch_size: int = 128, train: bool = True) -> DataLoader:
    """
    Creates a CIFAR-10 DataLoader for further training
    :param batch_size: batch size used in dataloader
    :param train: which part of dataset to return: either train or test
    :return: CIFAR-10 Dataloader with specified batch size
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean, std)
    ]) if train else T.Compose(
        [T.ToTensor(),
         T.Normalize(mean, std)
         ])

    dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader


def get_cifar100_dataloader(batch_size: int = 128, train: bool = True) -> DataLoader:
    """
        Creates a CIFAR-100 DataLoader for further training
        :param batch_size: batch size used in dataloader
        :param train: which part of dataset to return: either train or test
        :return: CIFAR-100 Dataloader with specified batch size
    """
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean, std)
    ]) if train else T.Compose(
        [T.ToTensor(),
         T.Normalize(mean, std)
         ])

    dataset = datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader
