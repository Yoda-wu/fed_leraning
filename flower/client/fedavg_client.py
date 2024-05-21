import time
from typing import Dict, Tuple

import flwr as fl
from collections import OrderedDict
from flwr.common.logger import log
from flwr.common import NDArrays, Scalar
from logging import INFO
import torch
import sys

sys.path.append('../..')
from flower.model.lenet5 import lenet5, train, test
from utils.config_parser import Parser
from flower.dataset.cifar10 import load_cifar10
from flower.dataset.mnist import load_mnist
import os

cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


class FedAvgClient(fl.client.NumPyClient):
    """
    参考Flower的示例，用户需要自己实现客户端的实体
    """

    def __init__(self, cid, net, trainloader, valloader, device):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device

    def set_parameters(self, parameters):
        """
        设置模型参数
        """
        try:
            log(INFO, f"in set parameters {len(parameters)}")
            param_dict = zip(self.net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in param_dict})
            self.net.load_state_dict(state_dict)
        except Exception as e:
            log(INFO, f"err occurs in set parameters {e}")

    def get_parameters(self, config):
        """
        获取模型参数
        """

        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters, config):
        """
        客户端本地训练过程
        """
        self.set_parameters(parameters)
        log(INFO, "in client fit process")
        epochs = config['local_epoch']
        start = time.time()
        train(self.net, self.trainloader, epochs, self.device)
        log(INFO, f"finish client fit process cause :{time.time() - start}s")
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(
            self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        客户端本地评估过程
        """
        pass


def gen_client_fn(trainloaders, valloaders, model, device):
    """
    生成client函数——供单机仿真测试使用
    """

    def client_fn(cid: str):
        return FedAvgClient(
            int(cid),
            model,
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            device=device
        )

    return client_fn


if __name__ == '__main__':

    args = Parser().parse()
    node_id = args.node_id
    client_number = int(args.client_number)
    batch_size = int(args.batch_size)
    Device = 'cpu'
    num_classes = 10
    model = None
    in_channels = 1
    load_data = None
    if args.dataset == 'mnist' or args.dataset == 'cifar10':
        num_classes = 10
    if args.dataset == 'mnist':
        load_data = load_mnist
        in_channels = 1
    if args.dataset == 'cifar10':
        load_data = load_cifar10
        in_channels = 3
    if args.model == 'lenet5':
        model = lenet5(in_channel=in_channels, num_classes=num_classes).to(Device)
    log(INFO, f"load_data {load_data}")
    trainloader, valloader = load_data(node_id, num_partitions=10, batch_size=batch_size,
                                       val_ratio=0)
    log(INFO, f"start client {node_id} !!!!!! ")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=FedAvgClient(node_id, model, trainloader, valloader, Device)
    )
