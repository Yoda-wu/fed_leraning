from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import CIFAR10
from flower.dataset.utils import load_data


def get_cifar10(data_path: str = "./data"):
    """
    获取CIFAR10数据集
    """
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    trainset = CIFAR10(data_path, train=True, download=True, transform=tr)
    testset = CIFAR10(data_path, train=False, download=True, transform=tr)
    return trainset, testset


def load_cifar10(node_id, num_partitions, batch_size, val_ratio, data_path: str = "./data"):
    """
    加载CIFAR10数据集
    """
    return load_data(get_cifar10, node_id, num_partitions, batch_size, val_ratio, data_path)
