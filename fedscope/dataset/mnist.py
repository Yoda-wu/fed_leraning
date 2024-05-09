import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
from federatedscope.register import register_data
from federatedscope.core.data import BaseDataTranslator


def load_mnist(config, client_cfgs=None):
    """
    这个文件是用来加载MNIST数据集的。
    加载MNIST并且注册到FederatedScope中。
    """
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    data_train = MNIST('../data', train=True, download=True, transform=tr)
    data_test = MNIST('../data', train=False, download=True, transform=tr)
    translator = BaseDataTranslator(config, client_cfgs)
    fs_data = translator((data_train, [], data_test))
    return fs_data, config


def call_mnist(config):
    if config.data.type == 'mnist':
        data, modified_config = load_mnist(config, None)
        return data, modified_config


register_data("mnist", call_mnist)
