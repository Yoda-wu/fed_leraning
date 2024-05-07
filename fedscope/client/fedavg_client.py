import logging
import pickle
import sys

from federatedscope.core.trainers import Context
from federatedscope.core.workers.base_client import BaseClient
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.communication import gRPCCommManager
from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.utils import calculate_time_cost, merge_dict_of_results
from federatedscope.register import register_worker
from federatedscope.core.auxiliaries.logging import logger

"""
FedScope的Client与FedML中的ClientManager类似，都是处理通信/事件的类。
用户只需要在BaseClient的框架下，实现默认的消息处理函数即可，也可以自定义消息处理函数。
本地训练由Trainer类完成，这里采用了FedScope提供了通用的TorchTrainer类，用户无需自己实现。
"""


class FedAvgClient(BaseClient):
    """
    FedAvg算法的客户端实现。
    本地训练过程偷懒使用了框架自带的GeneralTorchTrainer类。
    """
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):
        super(FedAvgClient, self).__init__(ID, state, config, model, strategy)
        self.trainer = None
        self.data = data
        self._register_default_handlers()
        self.is_unseen_client = is_unseen_client
        self.is_attacker = False
        logger.info(f'len{len(data)}  and   {data.keys()}')

        self.device = device
        self.best_results = dict()
        self.history_results = dict()
        self.msg_buffer = {'train': dict(), 'eval': dict()}
        self.server_id = server_id
        self.model_size = sys.getsizeof(pickle.dumps(
            self.model)) / 1024.0 * 8.
        self.comp_speed = None
        self.comm_bandwidth = None
        host = kwargs['host']
        port = kwargs['port']
        server_host = kwargs['server_host']
        server_port = kwargs['server_port']
        self.comm_manager = gRPCCommManager(
            host=host,
            port=port,
            client_num=self._cfg.federate.client_num,
            cfg=self._cfg.distribute)
        logger.info('Client: Listen to {}:{}...'.format(host, port))
        self.comm_manager.add_neighbors(neighbor_id=server_id,
                                        address={
                                            'host': server_host,
                                            'port': server_port
                                        })
        self.local_address = {
            'host': self.comm_manager.host,
            'port': self.comm_manager.port
        }
        logger.info(f'client cfg = {self._cfg.eval.split}')

    def _gen_timestamp(self, init_timestamp, instance_number):
        if init_timestamp is None:
            return None
        comp_cost, comm_cost = calculate_time_cost(
            instance_number=instance_number,
            comm_size=self.model_size,
            comp_speed=self.comp_speed,
            comm_bandwidth=self.comm_bandwidth)
        return init_timestamp + comp_cost + comm_cost

    def join_in(self):
        logger.info('client send join_in to server')
        self.comm_manager.send(
            Message(msg_type='join_in',
                    sender=self.ID,
                    receiver=[self.server_id],
                    timestamp=0,
                    content=self.local_address))

    def run(self):
        while True:
            msg = self.comm_manager.receive()
            if self.state <= msg.state:
                self.msg_handlers[msg.msg_type](msg)

            if msg.msg_type == 'finish':
                break

    def callback_funcs_for_model_para(self, message):
        """
        回调函数，处理来自server的model_para消息，即接收全局模型参数
        """
        round = message.state
        sender = message.sender
        logger.info(
            f"================= client {self.ID} received model_para message "
            f"=================")
        timestamp = message.timestamp
        content = message.content
        model_para = content[0]
        config = content[1]
        # logger.info(content)
        self.trainer.ctx['num_train_epoch'] = config[self.ID]['epochs']
        # 更新参数
        self.trainer.update(model_para, strict=self._cfg.federate.share_local_model)
        self.state = round
        # 本地训练
        sample_size, model_para_all, results = self.trainer.train()
        train_log_res = self._monitor.format_eval_res(results, rnd=self.state, role=f"client #{self.ID}",
                                                      return_raw=True)
        logger.info(train_log_res)
        shared_model_para = model_para_all
        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=self._gen_timestamp(
                        init_timestamp=timestamp,
                        instance_number=sample_size),
                    content=(sample_size, shared_model_para)))

    def callback_funcs_for_assign_id(self, message):
        """
        接收到server的assign_id消息，即分配client ID
        """
        content = message.content
        self.ID = int(content)
        data = self.data[self.ID]
        logger.info(f"{len(data)}, {len(data['train'])} {data['train'].batch_size}")
        logger.info(
            f"================= client {self.ID} received assign_id message "
            f"=================")
        self.trainer = GeneralTorchTrainer(model=self.model,
                                           data=data,
                                           device=self.device,
                                           config=self._cfg,
                                           only_for_eval=False,
                                           monitor=self._monitor)
        logger.info(self.trainer)
        logger.info(f"client (address {self.comm_manager.host}:{self.comm_manager.port}) is assigned with {self.ID}")
        self.comm_manager.send(
            Message(msg_type='confirm_assign_id', sender=self.ID, receiver=[self.server_id], state=self.state)
        )

    def callback_funcs_for_join_in_info(self, message):
        """
        接收到server的join_in_info消息，获取本地训练所需的样本数
        """
        logger.info(
            f"================= client {self.ID} received join_in_info message "
            f"=================")
        requirements = message.content
        timestamp = message.timestamp
        join_in_info = dict()

        for requirement in requirements:
            if requirement.lower() == 'num_sample':
                num_sample = self._cfg.train.local_update_steps * \
                             len(self.trainer.data.train_data)
                join_in_info['num_sample'] = num_sample
        self.comm_manager.send(
            Message(msg_type='join_in_info',
                    sender=self.ID,
                    receiver=[self.server_id],
                    state=self.state,
                    timestamp=timestamp,
                    content=join_in_info))

    def callback_funcs_for_address(self, message):
        logger.info(
            f"================= client {self.ID} received address message "
            f"=================")
        content = message.content
        for neighbor_id, address in content.items():
            if int(neighbor_id) != self.ID:
                self.comm_manager.add_neighbors(neighbor_id, address)

    def callback_funcs_for_evaluate(self, message):
        logger.info(
            f"================= client {self.ID} received evaluate message "
            f"=================")
        sender, timestamp, content = message.sender, message.timestamp, message.content
        self.state = message.state
        if content is not None:
            self.trainer.update(content, strict=self._cfg.federate.share_local_model)
        metrics = {}
        for split in self._cfg.eval.split:
            eval_metrics = self.trainer.evaluate(
                target_data_split_name=split
            )
            logger.info(
                self._monitor.format_eval_res(eval_metrics,
                                              rnd=self.state,
                                              role='Client #{}'.format(
                                                  self.ID),
                                              return_raw=True))
            metrics.update(**eval_metrics)

        formatted_eval_res = self._monitor.format_eval_res(
            metrics,
            rnd=self.state,
            role='Client #{}'.format(self.ID),
            forms=['raw'],
            return_raw=True)
        logger.info(f'eval res :{formatted_eval_res}')

        self._monitor.update_best_result(self.best_results,
                                         formatted_eval_res['Results_raw'],
                                         results_type=f"client #{self.ID}")
        self.history_results = merge_dict_of_results(
            self.history_results, formatted_eval_res['Results_raw'])

    def callback_funcs_for_finish(self, message):
        logger.info(
            f"================= client {self.ID} received finish message "
            f"=================")
        if message.content is not None:
            self.trainer.update(message.content,
                                strict=self._cfg.federate.share_local_model)

        self._monitor.finish_fl()

    def callback_funcs_for_converged(self, message):
        self._monitor.global_converged()
