from typing import Dict, Tuple

import flwr as fl
from collections import OrderedDict

from flwr.common import NDArrays, Scalar

from flower.model.lenet5 import train, test

import torch


class FedAvgClient(fl.client.NumPyClient):
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
            print(f"err occurs in set parameters {e}")

    def get_parameters(self, config):
        return [val.cpu.numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        epochs = config['local_epochs']
        train(self.net, self.trainloader, epochs, self.device)
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(
            self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        loss, acc = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"acc": acc}


def gen_client_fn(trainloaders, valloaders, model, device):
    def client_fn(cid: str):
        return FedAvgClient(
            int(cid),
            model,
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            device=device
        )
    return client_fn