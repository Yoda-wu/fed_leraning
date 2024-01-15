import logging
import sys
import copy

sys.path.append('..')
sys.path.append('../..')
from fedscale.cloud.aggregation.aggregator import Aggregator
from fedscale.cloud.internal.torch_model_adapter import TorchModelAdapter
from fed_scale.model.lenet5 import LeNet5
import fedscale.cloud.config_parser as parser
from fed_scale.client_manager import FedAvgClientManager


class FedAvgAggregator(Aggregator):
    """
    FedAvg算法的聚合器实现，继承自fedscale.cloud.aggregation.aggregator.Aggregator类以屏蔽对底层通信的感知，来实现算法的聚合逻辑
    这里即充当server角色又充当aggregator角色.
    聚合功能就在update_weight_aggregation这个函数中实现。
    客户端选择功能则在client_manager中实现。

    """

    def __init__(self, args):
        super(FedAvgAggregator, self).__init__(args)
        self.client_select_dict = dict()

        self.client_manager = self.init_client_manager(args)

    def init_client_manager(self, args):
        client_manager = FedAvgClientManager(args.sample_mode , args=args)
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
        # super().update_weight_aggregation(results)
        update_weights = results["update_weight"]  # 这里传输的权重以及是加权过后的 i.e. w[i] = w[i] * (data[i] / total_data)
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


if __name__ == '__main__':
    aggregator = FedAvgAggregator(parser.args)
    aggregator.run()
