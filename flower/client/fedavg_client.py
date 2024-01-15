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
from fed_scale.model.lenet5 import lenet5, train, test
from utils.config_parser import Parser
from flower.dataset.mnist import load_data


class FedAvgClient(fl.client.NumPyClient):
    """
    FedAvg算法的客户端实现，继承自flower框架里的NumPyClient类
    本地训练的功能由这个类实现
    功能有：
    1.本地模型训练
    2.本地模型评估
    3.获取本地模型参数
    4.设置本地模型参数

    """
    def __init__(self, cid, net, trainloader, valloader, device):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device

    def set_parameters(self, parameters):
        try:
            param_dict = zip(self.net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in param_dict})
            self.net.load_state_dict(state_dict)
        except Exception as e:
            log(INFO, f"err occurs in set parameters {e}")

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
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
        self.set_parameters(parameters)
        loss, acc = test(self.net, self.valloader, self.device)
        log(INFO, f"client-side evaluation loss {loss} / accuracy {acc}")
        return float(loss), len(self.valloader), {"acc": acc}


def gen_client_fn(trainloaders, valloaders, model, device):
    """
    生成客户端的函数，用于创建客户端实例，这个方法会有flower框架调用
    启动客户端的时候，会调用这个方法，生成客户端实例。
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


def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


if __name__ == '__main__':

    args = Parser().parse()
    node_id =args.node_id
    client_number = int(args.client_number)
    batch_size = int(args.batch_size)
    Device = 'cpu'
    num_classes = 1
    model = None
    if args.dataset == 'mnist':
        num_classes = 10
    if args.model == 'lenet5' :
        model = lenet5(num_classes=num_classes).to(Device)
    trainloader, valloader = load_data(node_id, client_number,batch_size, val_ratio=0.3)
    log(INFO, f"start client {node_id} !!!!!! ")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=FedAvgClient(node_id, model, trainloader, valloader, Device)
    )