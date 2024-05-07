from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
from flower.dataset.utils import load_data


def get_mnist(data_path: str = "./data"):
    """
    获取MNIST数据集
    """
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)
    return trainset, testset


def load_mnist(node_id, num_partitions, batch_size, val_ratio, data_path: str = "./data"):
    """
    加载CIFAR10数据集
    """
    return load_data(get_mnist, node_id, num_partitions, batch_size, val_ratio, data_path)
