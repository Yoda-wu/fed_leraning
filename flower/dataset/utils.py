import torch
from torch.utils.data import random_split, DataLoader


def prepare_dataset(get_data_fn, num_partitions, batch_size, val_ratio=0.3, data_path: str = "./data"):
    """
    数据集准备，将数据集分为num_partitions份，其中val_ratio比例的数据用于验证
    :param get_data_fn: 获取数据集的函数
    :param num_partitions: 分为num_partitions份
    :param batch_size: batch大小
    :param val_ratio: 验证集比例
    :param data_path: 数据集路径
    :return: trainloader, valloader, testloader
    """
    trainset, testset = get_data_fn(data_path)

    num_images = len(trainset) // num_partitions
    partition_len = [num_images] * num_partitions

    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(42))
    trainloader = []
    valloader = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val
        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(42))
        trainloader.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=0))
        valloader.append(DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=0))

    testloader = DataLoader(testset, batch_size=128)
    return trainloader, valloader, testloader


def load_data(get_data_fn, node_id, num_partitions, batch_size, val_ratio, data_path: str = "./data"):
    """
    加载数据集
    """
    trainloaders, valloaders, testloader = prepare_dataset(get_data_fn, num_partitions, batch_size, val_ratio, data_path)
    return trainloaders[node_id], valloaders[node_id]