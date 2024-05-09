import flwr as fl
from config import Configuration
from strategy.fed_avg import FedAvgStrategy
import sys
import torch
from flower.model.lenet5 import test
from client.fedavg_client_manager import FedAvgClientManager
from dataset.utils import prepare_dataset
from dataset.mnist import get_mnist
from dataset.cifar10 import get_cifar10
from flwr.common.logger import log
from logging import INFO
from collections import OrderedDict
from flwr.common.typing import Dict, Tuple, Optional

sys.path.append('../')
print(sys.path)
from utils.config_parser import Parser

fl.common.logger.configure(identifier="myFlowerExperiment", filename="log.txt")

# The `evaluate` function will be by Flower called after every round
args = Parser().parse()

Config = Configuration(args)
Config.device = 'cpu'
Config.generate_config_dict()
Config.show_configuration()
_, _, testloader = [], [], []
if Config.dataset == 'mnist':
    _, _, testloader = prepare_dataset(get_mnist, Config.client_number, Config.batch_size,
                                       val_ratio=0.3)
    print('finish loading dataset')
elif Config.dataset == 'cifar10':
    _, _, testloader = prepare_dataset(get_cifar10, Config.client_number, Config.batch_size,
                                       val_ratio=0.3)
strategy: FedAvgStrategy = None


def get_parameters(net):
    """
    获取模型参数
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    """
    全局评估过程，作为strategy里的evaluate_fn
    """
    net = Config.model
    valloader = testloader
    try:
        param_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in param_dict})
        net.load_state_dict(state_dict)
    except Exception as e:
        log(INFO, f"err occurs in set parameters {e}")  # Update model with the latest parameters
    loss, accuracy = test(net, valloader, Config.device)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}

    # if args.gpu == None:
    #     Config.device = 'cpu'
    # else:
    #     Config.device = 'cuda'


if Config.frac_clients != 0:
    print('choose frac clients to select')
    strategy = FedAvgStrategy(
        fraction_fit=Config.frac_clients,  # Sample 100% of available clients for training
        fraction_evaluate=Config.frac_clients,  # Sample 50% of available clients for evaluation
        min_fit_clients=int(Config.frac_clients * Config.client_number),
        # Never sample less than 10 clients for training
        min_evaluate_clients=int(Config.frac_clients * Config.client_number),
        # Never sample less than 5 clients for evaluation
        min_available_clients=int(Config.frac_clients * Config.client_number),
        # Wait until all 10 clients are available
        available_clients=int(Config.frac_clients * Config.client_number),
        evaluate_fn=evaluate,
        model=Config.model,
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Config.model)),
    )
else:
    print('choose available clients to select')
    rate = float(Config.available_clients) / float(Config.client_number)
    strategy = FedAvgStrategy(
        fraction_fit=rate,  # Sample 100% of available clients for training
        fraction_evaluate=rate,  # Sample 50% of available clients for evaluation
        min_fit_clients=Config.available_clients,  # Never sample less than 10 clients for training
        min_evaluate_clients=Config.available_clients,
        # Never sample less than 5 clients for evaluation
        min_available_clients=Config.available_clients,  # Wait until all 10 clients are available
        available_clients=Config.available_clients,
        model=Config.model
    )
assert strategy is not None
print(f'finish create strategy {strategy.__repr__()} {strategy.fraction_fit}')
fl.server.start_server(
    server_address=Config.server_address,
    config=Config.server_config,
    strategy=strategy,
    client_manager=FedAvgClientManager()
)
