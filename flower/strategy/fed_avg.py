import numpy as np
from functools import reduce
from flwr.common.logger import log
from logging import INFO
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import sys

sys.path.append('..')


class FedAvgStrategy(Strategy):
    """
    Flower中实现联邦学习的核心部分——Strategy，定义了服务端的主要行为。
    用户需要自己实现对应的方法
    """

    def __init__(self,
                 fraction_fit: float = 1.0,
                 fraction_evaluate: float = 1.0,
                 min_fit_clients: int = 2,
                 min_evaluate_clients: int = 2,
                 min_available_clients: int = 2,
                 available_clients: int = 2,
                 model=None,
                 valdata=None,
                 evaluate_fn: Optional[
                     Callable[
                         [int, NDArrays, Dict[str, Scalar]],
                         Optional[Tuple[float, Dict[str, Scalar]]],
                     ]
                 ] = None,
                 initial_parameters: Optional[Parameters] = None,
                 on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
                 on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
                 ):

        self.initial_parameters = initial_parameters
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.available_clients = available_clients
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.client_dict = {}  # cid -> client
        self.model = model
        self.valdata = valdata

    def __repr__(self):
        rep = f"FedAvg_Custom"
        return rep

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager: ClientManager) -> List[
        Tuple[ClientProxy, FitIns]]:
        log(INFO, "--------------in configure_fit------------------")
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        log(INFO, f"clients {clients}")
        fit_config = []
        initial_config = {'local_epoch': 2}
        standard_config = {'local_epoch': 1}
        for client in clients:
            if client.cid not in self.client_dict.keys():
                self.client_dict[client.cid] = 1
            else:
                self.client_dict[client.cid] = self.client_dict[client.cid] + 1
            if self.client_dict[client.cid] < 3:
                fit_config.append((client, FitIns(parameters, initial_config)))
            else:
                fit_config.append((client, FitIns(parameters, standard_config)))

        return fit_config

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[
        Optional[Parameters], Dict[str, Scalar]]:
        """
        FedAvg聚合操作
        """
        if not results:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in
            results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate_helper(weights_results))
        return parameters_aggregated, {}

    def configure_evaluate(self, server_round: int, parameters: Parameters,
                           client_manager: ClientManager) -> List[
        Tuple[ClientProxy, EvaluateIns]]:
        pass


    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]) -> \
            Tuple[
                Optional[float], Dict[str, Scalar]]:
        pass

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[
        Tuple[float, Dict[str, Scalar]]]:
        """
        服务端全局评估过程
        """
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        log(INFO, f"server-side evaluation loss {loss} / accuracy {metrics['accuracy']}")
        return loss, metrics

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients


def aggregate_helper(results):
    """
    FedAvg聚合操作的辅助函数，即对于每个客户端的更新参数进行加权平均
    """
    num_examples_total = sum([num_examples for _, num_examples in results])

    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]
    weights_prim = [
        reduce(np.add, layer_updates) / num_examples_total for layer_updates in
        zip(*weighted_weights)
    ]
    return weights_prim


def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """
    平均多个客户端的评估结果
    """
    num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples
