import logging
import sys
import copy
import time

sys.path.append('..')
sys.path.append('../..')

from argparse import Namespace
from fedscale.cloud.aggregation.aggregator import Aggregator
from fedscale.cloud.internal.torch_model_adapter import TorchModelAdapter
from fed_scale.model.lenet5 import LeNet5
import fedscale.cloud.config_parser as parser
from fed_scale.client_manager import FedAvgClientManager
from fed_scale.execution.trainer import FedAvgTrainer
from fedscale.dataloaders.divide_data import DataPartitioner, select_dataset
from fedscale.cloud.fllibs import *


class FedAvgAggregator(Aggregator):
    """
    FedAvg的聚合器， 负责FedAvg中服务端所有功能。
    """

    def __init__(self, args):
        super(FedAvgAggregator, self).__init__(args)
        self.testing_data_loader = None
        self.client_select_dict = dict()
        self.client_manager = self.init_client_manager(args)
        self.trainer = FedAvgTrainer(args)
        self.begin_timer = None
        self.init_test_data()

    def init_test_data(self):
        _, testing_data = init_dataset()
        testing_data = DataPartitioner(
            data=testing_data,
            args=self.args,
            numOfClass=self.args.num_class,
            isTest=True,
        )
        testing_data.partition_data_helper(num_clients=1, data_map_file=None)
        testing_data_loader = select_dataset(0, testing_data, self.args.batch_size, self.args,
                                             isTest=True)
        self.testing_data_loader = testing_data_loader

    def round_completion_handler(self):
        super(FedAvgAggregator, self).round_completion_handler()
        if self.round == 1:
            self.begin_timer = time.time()

    def init_client_manager(self, args):
        # client_manager负责客户端选择
        client_manager = FedAvgClientManager(args.sample_mode, args=args)
        return client_manager

    def init_model(self):
        model = LeNet5(10)
        self.model_wrapper = TorchModelAdapter(
            model,
            optimizer=None
        )

    def get_client_conf(self, client_id):
        if client_id not in self.client_select_dict.keys():
            self.client_select_dict[client_id] = 1
        self.client_select_dict[client_id] += 1

        conf = {
            "learning_rate": self.args.learning_rate,
            "epochs": 1 if self.client_select_dict[client_id] > 2 else 2
        }
        return conf

    def update_weight_aggregation(self, results):
        logging.info(self.tasks_round)
        update_weights = results["update_weight"]
        # 这里传输的权重以及是加权过后的 i.e. w[i] = w[i] * (data[i] / total_data)
        # 因此下面只需要简单的相加即可
        if type(update_weights) is dict:
            update_weights = [x for x in update_weights.values()]
        if self._is_first_result_in_round():
            self.model_weights = update_weights
        else:
            self.model_weights = [
                weight + update_weights[i]
                for i, weight in enumerate(self.model_weights)
            ]
        if self._is_last_result_in_round():
            self.model_wrapper.set_weights(
                copy.deepcopy(self.model_weights),
                client_training_results=self.client_training_results,
            )
            self.test_on_server()
            if self.round == self.args.rounds - 1:
                logging.info(f"Time used: {time.time() - self.begin_timer}")

    def override_conf(self, config):
        """Override the variable arguments for different client

        Args:
            config (dictionary): The client runtime config.

        Returns:
            dictionary: Variable arguments for client runtime config.

        """
        default_conf = vars(self.args).copy()

        for key in config:
            default_conf[key] = config[key]

        return Namespace(**default_conf)

    def test_on_server(self):
        test_config = self.override_conf(
            {
                "rank": self.this_rank,
                "memory_capacity": self.args.memory_capacity,
                "tokenizer": tokenizer,
            }
        )
        test_results = self.trainer.test(
            self.testing_data_loader, model=self.model_wrapper.get_model(), conf=test_config
        )

        logging.info(
            f"------Test on server results round {self.round}/{self.args.rounds}: test_acc： {test_results['top_1'] / test_results['test_len']} {test_results}")

    def testing_completion_handler(self, client_id, results):
        """Each executor will handle a subset of testing dataset

        Args:
            client_id (int): The client id.
            results (dictionary): The client test results.

        """

        results = results["results"]

        # List append is thread-safe
        self.test_result_accumulator.append(results)

        # Have collected all testing results

        if len(self.test_result_accumulator) == len(self.executors):
            self.aggregate_test_result()
            if len(self.loss_accumulator):
                logging.info("logging test result")
                self.log_test_result()

            self.broadcast_events_queue.append(commons.START_ROUND)


if __name__ == '__main__':
    aggregator = FedAvgAggregator(parser.args)

    logging.info(f"args: {parser.args}")
    aggregator.run()
