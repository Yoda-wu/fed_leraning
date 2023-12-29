import torch
from torch.utils.data import  random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST

def get_mnist(data_path:str = "./data"):
    tr = Compose([ToTensor(), Normalize((0.1307,),(0.3081, ))])

    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)
    return trainset, testset

