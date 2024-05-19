from federatedscope.core.data import BaseDataTranslator
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import CIFAR10
from federatedscope.register import register_data


def load_cifar10(config, client_cfgs=None):
    """
    获取CIFAR10数据集
    """
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    print("load cifar10!!!")
    data_train = CIFAR10("../data", train=True, download=True, transform=tr)
    data_test = CIFAR10("../data", train=False, download=True, transform=tr)
    translator = BaseDataTranslator(config, client_cfgs)
    fs_data = translator((data_train, [], data_test))
    return fs_data, config


def call_cifar10(config):
    if config.data.type == 'cifar10':
        data, modified_config = load_cifar10(config, None)
        return data, modified_config


register_data("cifar10", load_cifar10)
