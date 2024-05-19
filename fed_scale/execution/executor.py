import collections
import logging
import os
import time
import sys
import fedscale.cloud.config_parser as parser
import wandb
import torch
from fedscale.cloud.execution.executor import Executor
from fedscale.cloud.channels.channel_context import ClientConnections
import fedscale.cloud.logger.executor_logging as logger

sys.path.append('..')
sys.path.append('../..')
from fed_scale.model.lenet5 import LeNet5
import os

cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


class FedAvgExecutor(Executor):
    """
    负责启动client， 并且执行client的训练任务
    """

    def __init__(self, args):
        print(f"Executor init data {parser.args.data_set}")
        self.model_adapter = self.get_client_trainer(args).get_model_adapter(
            LeNet5(10)
        )
        # 由于FedScale的模型库不支持LeNet，并且也不支持用户自定义模型，所以这里将Executor的初始化copy过来了。
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
        logging.info(f"num of class {self.args.num_class}")

    def Train(self, config):
        """Load train config and data to start training on that client

        Args:
            config (dictionary): The client training config.

        Returns:
            tuple (int, dictionary): The client id and train result

        """
        logging.info(self.report_executor_info_handler())
        config["epochs"] = int(config['task_config']["epochs"])
        total_data = 0
        sizes = self.training_sets.getSize()['size']
        for size in sizes:
            total_data += size
        config['task_config']['total_data'] = total_data
        client_id, train_res = super().Train(config)
        return client_id, train_res

    def get_client_trainer(self, conf):
        from trainer import FedAvgTrainer
        return FedAvgTrainer(conf)


if __name__ == '__main__':
    print("hello!!!")
    executor = FedAvgExecutor(parser.args)
    executor.run()
