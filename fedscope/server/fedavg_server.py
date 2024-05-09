import sys

from fedscope.server.sampler import RandomSampler

sys.path.append('../..')
sys.path.append('.')
from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.logging import logger
from federatedscope.core.workers.server import Server
from fedscope.server.aggregator import FedAvgAggregator

"""
FedScope的Server与FedML中的ServerManager类似，都是处理通信/事件的类。
用户只需要在BaseClient的框架下，实现默认的消息处理函数即可，也可以自定义消息处理函数。
而聚合操作，这里采用自己实现的FedAvgAggregator类。
"""


class FedAvgServer(Server):
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 unseen_clients_id=None,
                 **kwargs):

        super(FedAvgServer, self).__init__(ID=ID,
                                           state=state,
                                           config=config,
                                           data=data,
                                           model=model,
                                           client_num=client_num,
                                           total_round_num=total_round_num,
                                           device=device,
                                           strategy=strategy,
                                           unseen_clients_id=unseen_clients_id, **kwargs
                                           )
        self.join_in_client_id = 0
        self._register_default_handlers()
        self.model = model
        self.data = data
        self.device = device
        self.best_results = dict()
        self.history_results = dict()
        self.aggregator = FedAvgAggregator(model=model, device=device, config=config)
        self.client_select_dict = dict()
        self.setting_round = 2
        self.models = [self.model]
        self.model_num = config.model.model_num_per_trainer or len(self.models)
        self.aggregators = [self.aggregator]
        self.sampler = RandomSampler(
            client_num=self.client_num
        )

    def _register_default_handlers(self):
        super(FedAvgServer, self)._register_default_handlers()
        self.register_handlers("confirm_assign_id", self.callback_funcs_for_confirm_assign_id,
                               ['confirm_assign_id'])

    def callback_funcs_for_confirm_assign_id(self, message):
        """
        这里确保每个client都有一个唯一的ID，才会开始启动训练。
        """
        sender = message.sender
        logger.info(
            f"server {self.ID} received confirm_assign_id message from {sender} "
        )
        self.join_in_client_num += 1
        self.trigger_for_start()

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        config = dict()
        if sample_client_num > 0:
            receiver = self.sampler.sample(size=sample_client_num)
        else:
            receiver = list(self.comm_manager.neighbors.keys())
        logger.info(f'receiver is {receiver}')
        if msg_type == 'model_para':
            self.sampler.change_state(receiver, 'working')
            for client_id in receiver:
                config[client_id] = dict()
                self.client_select_dict[client_id] = self.state
                if self.client_select_dict[client_id] < self.setting_round:
                    config[client_id]["epochs"] = 2
                else:
                    config[client_id]["epochs"] = 1
        skip_broadcast = self._cfg.federate.method in ['local', 'global']
        model_para = {} if skip_broadcast else self.model.state_dict()
        rnd = self.state - 1 if msg_type == 'evaluate' else self.state
        if msg_type == 'model_para':
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=receiver,
                        state=min(rnd, self.total_round_num),
                        timestamp=self.cur_timestamp,
                        content=(model_para, config))
            )
        else:
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=receiver,
                        state=min(rnd, self.total_round_num),
                        timestamp=self.cur_timestamp,
                        content=model_para)
            )

    def _start_new_training_round(self, aggregated_num=0):
        self.broadcast_model_para(msg_type='model_para', sample_client_num=self.sample_client_num)

    def trigger_for_start(self):
        logger.info(
            f'----------- in trigger for start training #{self.check_client_join_in()} -------------')
        if self.check_client_join_in():
            self.trigger_for_feat_engr(
                self.broadcast_model_para, {
                    'msg_type': 'model_para',
                    'sample_client_num': self.sample_client_num
                })
            logger.info(f'----------- Starting training (Round #{self.state}) -------------')

    def callback_funcs_for_join_in(self, message):
        """
        处理客户端加入训练的消息
        """
        if 'info' in message.msg_type:
            # callback for join_in_info
            sender, info = message.sender, message.content
            for key in self._cfg.federate.join_in_info:
                assert key in info
            self.join_in_info[sender] = info
            logger.info(f'server: client {sender} has joined in !')
        else:
            self.join_in_client_id += 1
            sender, address = message.sender, message.content
            if int(sender) == -1:
                sender = self.join_in_client_id
                self.client_select_dict[sender] = self.state
                self.comm_manager.add_neighbors(neighbor_id=sender, address=address)
                self.comm_manager.send(
                    Message(msg_type='assign_client_id',
                            sender=self.ID,
                            receiver=[sender],
                            state=self.state,
                            timestamp=self.cur_timestamp,
                            content=str(sender)))
                logger.info(
                    f'server: new client {sender} has joined in ! now has {self.join_in_client_num} client in clusters')
            else:
                self.comm_manager.add_neighbors(neighbor_id=sender, address=address)
            if len(self._cfg.federate.join_in_info) != 0:
                self.comm_manager.send(
                    Message(msg_type='ask_for_join_in_info',
                            sender=self.ID,
                            receiver=[sender],
                            state=self.state,
                            timestamp=self.cur_timestamp,
                            content=self._cfg.federate.join_in_info.copy()))

    def run(self):
        while self.join_in_client_num < self.client_num:
            logger.info('waiting client to join....')
            msg = self.comm_manager.receive()
            self.msg_handlers[msg.msg_type](msg)
        while self.state <= self.total_round_num:
            msg = self.comm_manager.receive()
            move_on_flag = self.msg_handlers[msg.msg_type](msg)
        self.terminate(msg_type='finish')
