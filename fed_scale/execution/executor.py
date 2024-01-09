import time
import wandb
import collections
import logging
import os
from fedscale.cloud.channels.channel_context import ClientConnections
from fedscale.cloud.execution.executor import Executor
import fedscale.cloud.config_parser as parser
import fedscale.cloud.logger.executor_logging as logger
import sys

sys.path.append('..')
sys.path.append('../..')
from fed_scale.model.lenet5 import LeNet5


class FedAvgExecutor(Executor):
    def __init__(self, args):
        logger.initiate_client_setting()

        self.model_adapter = self.get_client_trainer(args).get_model_adapter(
            LeNet5(10)
        )

        self.args = args
        self.num_executors = args.num_executors
        # ======== env information ========
        self.this_rank = args.this_rank
        self.executor_id = str(self.this_rank)

        # ======== model and data ========
        self.training_sets = self.test_dataset = None

        # ======== channels ========
        self.aggregator_communicator = ClientConnections(args.ps_ip, args.ps_port)

        # ======== runtime information ========
        self.collate_fn = None
        self.round = 0
        self.start_run_time = time.time()
        self.received_stop_request = False
        self.event_queue = collections.deque()

        if args.wandb_token != "":
            os.environ["WANDB_API_KEY"] = args.wandb_token
            self.wandb = wandb
            if self.wandb.run is None:
                self.wandb.init(
                    project=f"fedscale-{args.job_name}",
                    name=f"executor{args.this_rank}-{args.time_stamp}",
                    group=f"{args.time_stamp}",
                )
            else:
                logging.error("Warning: wandb has already been initialized")

        else:
            self.wandb = None
        super(Executor, self).__init__()

    def Train(self, config):
        """Load train config and data to start training on that client

        Args:
            config (dictionary): The client training config.

        Returns:
            tuple (int, dictionary): The client id and train result

        """
        # logging.info(config)
        config["epochs"] = int(config['task_config']["epochs"])
        total_data = 0
        sizes = self.training_sets.getSize()['size']
        for size in sizes:
            total_data += size
        config['task_config']['total_data']=total_data
        client_id, train_res = super().Train(config)
        return client_id, train_res

    def get_client_trainer(self, conf):
        from trainer import FedAvgTrainer
        return FedAvgTrainer(conf)


if __name__ == '__main__':
    executor = FedAvgExecutor(parser.args)
    executor.run()
