import copy
import logging
import pickle
import sys
import threading

sys.path.append('../..')
sys.path.append('.')
from federatedscope.core.communication import gRPCCommManager
from federatedscope.core.message import Message
from federatedscope.core.workers.base_server import BaseServer
from federatedscope.core.auxiliaries.aggregator_builder import get_aggregator
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from fedscope.server.sampler import RandomSampler
from federatedscope.core.auxiliaries.utils import Timeout, merge_param_dict, merge_dict_of_results
from federatedscope.core.auxiliaries.logging import logger


class FedAvgServer(BaseServer):
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

        super(FedAvgServer, self).__init__(ID, state, config, model, strategy)

        self._register_default_handlers()
        self.model = model
        self.data = data
        self.device = device
        self.best_results = dict()
        self.history_results = dict()
        self.aggregator = get_aggregator(self._cfg.federate.method,
                                         model=model,
                                         device=device,
                                         online=self._cfg.federate.online_aggr,
                                         config=self._cfg)

        self.models = [self.model]
        self.model_num = config.model.model_num_per_trainer or len(self.models)
        self.aggregators = [self.aggregator]

        if self._cfg.federate.make_global_eval:
            import torch.nn as nn
            assert self.models is not None
            assert self.data is not None
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = False
            self.trainer = get_trainer(
                model=self.model,
                data=self.data[0],
                device=self.device,
                config=self._cfg,
                only_for_eval=True,
                monitor=self._monitor
            )  # the trainer is only used for global evaluation
            self.trainers = [self.trainer]
        # Initialize the number of joined-in clients
        self.client_num = client_num
        self.total_round_num = total_round_num
        self.sample_client_num = int(self._cfg.federate.sample_client_num)
        self.join_in_client_id = 0
        self.join_in_client_num = 0
        self.join_in_info = dict()
        self.dropout_num = 0
        self.sampler = RandomSampler(
            client_num=self.client_num
        )
        self.lock = threading.Lock()
        self.client_select_dict = dict()
        self.setting_round = 2
        self.is_finish = False
        # Device information
        self.resource_info = kwargs['resource_info'] \
            if 'resource_info' in kwargs else None
        self.client_resource_info = kwargs['client_resource_info'] \
            if 'client_resource_info' in kwargs else None
        self.cur_timestamp = 0
        self.deadline_for_cur_round = 1
        self.msg_buffer = {'train': dict(), 'eval': dict()}
        host = kwargs['host']
        port = kwargs['port']
        self.comm_manager = gRPCCommManager(host=host,
                                            port=port,
                                            client_num=client_num,
                                            cfg=self._cfg.distribute)
        logger.info('Server: Listen to {}:{}...'.format(host, port))

    def _register_default_handlers(self):
        super()._register_default_handlers()
        self.register_handlers("confirm_assign_id", self.callback_funcs_for_confirm_assign_id, ['confirm_assign_id'])

    def callback_funcs_for_confirm_assign_id(self, message):
        sender = message.sender
        logger.info(
            f"server {self.ID} received confirm_assign_id message from {sender} "
        )
        self.join_in_client_num += 1
        self.trigger_for_start()

    def check_buffer(self, cur_round, min_received_num=None, check_eval_result=False):
        if check_eval_result:
            if 'eval' not in self.msg_buffer.keys() or len(self.msg_buffer['eval'].keys()) == 0:
                return False
            buffer = self.msg_buffer['eval']
            cur_round = max(buffer.keys())
            cur_buffer = buffer[cur_round]
            return len(cur_buffer) >= min_received_num
        else:
            if cur_round not in self.msg_buffer['train']:
                cur_buffer = dict()
            else:
                cur_buffer = self.msg_buffer['train'][cur_round]
            return len(cur_buffer) >= min_received_num

    def _perform_federated_aggregation(self):
        train_msg_buffer = self.msg_buffer['train'][self.state]
        model = self.model
        aggregator = self.aggregator
        msg_list = list()
        for client_id in train_msg_buffer:
            msg_list.append(train_msg_buffer[client_id])

        self._monitor.calc_model_metric(self.model.state_dict(), msg_list, rnd=self.state)
        agg_info = {
                'client_feedback': msg_list,
                'recover_fun': None,
                'staleness': []
        }
        result = aggregator.aggregate(agg_info)
        merged_param = merge_param_dict(model.state_dict().copy(), result)
        model.load_state_dict(merged_param, strict=False)
        self.model = model
        self.trainer.update(merged_param, strict=False)

    def eval(self):
        if self._cfg.federate.make_global_eval:
            # 全局评估
            trainer = self.trainer
            metrics = {}
            for split in self._cfg.eval.split:
                eval_metrics = trainer.evaluate(target_data_split_name=split)
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
                role='Server #',
                forms=self._cfg.eval.report,
                return_raw=self._cfg.federate.make_global_eval)
            self._monitor.update_best_result(
                self.best_results,
                formatted_eval_res['Results_raw'],
                results_type="server_global_eval")
            self.history_results = merge_dict_of_results(
                self.history_results, formatted_eval_res)
            self._monitor.save_formatted_results(formatted_eval_res)
            logger.info(f'------global eval res :{formatted_eval_res}-------')
        else:
            # 本地评估
            self.broadcast_model_para(msg_type='evaluate')

    def broadcast_model_para(self, msg_type, sample_client_num=-1):
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


    def _start_new_training_round(self):
        self.broadcast_model_para(msg_type='model_para', sample_client_num=self.sample_client_num)

    def merge_eval_results_from_all_clients(self):
        round = max(self.msg_buffer['eval'].keys())
        eval_msg_buffer = self.msg_buffer['eval'][round]
        eval_res_participated_clients = []
        for client_id in eval_msg_buffer:
            if eval_msg_buffer[client_id] is None:
                continue
            eval_res_participated_clients.append(eval_msg_buffer[client_id])
        formatted_logs_all_set = dict()
        metrics_all_clients = dict()
        for client_eval_results in eval_res_participated_clients:
            for key in client_eval_results.keys():
                if key not in metrics_all_clients:
                    metrics_all_clients[key] = list()
                metrics_all_clients[key].append(
                    float(client_eval_results[key]))
        formatted_logs = self._monitor.format_eval_res(
            metrics_all_clients,
            rnd=round,
            role='Server #',
            forms=self._cfg.eval.report)
        logger.info(formatted_logs)
        formatted_logs_all_set.update(formatted_logs)
        self._monitor.update_best_result(
            self.best_results,
            metrics_all_clients,
            results_type="client_best_individual")
        self._monitor.save_formatted_results(formatted_logs)
        for form in self._cfg.eval.report:
            if form != "raw":
                metric_name = form + form
                self._monitor.update_best_result(
                    self.best_results,
                    formatted_logs[f"Results_{metric_name}"],
                    results_type=f"client_summarized_{form}")
        return formatted_logs_all_set

    def save_best_results(self):
        """
        To Save the best evaluation results.
        """

        if self._cfg.federate.save_to != '':
            self.aggregator.save_model(self._cfg.federate.save_to, self.state)
        formatted_best_res = self._monitor.format_eval_res(
            results=self.best_results,
            rnd="Final",
            role='Server #',
            forms=["raw"],
            return_raw=True)
        logger.info(formatted_best_res)
        self._monitor.save_formatted_results(formatted_best_res)

    def terminate(self, msg_type='finish'):
        self.is_finish = True
        model_para = self.model.state_dict()
        self._monitor.finish_fl()
        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=list(self.comm_manager.neighbors.keys()),
                    state=self.state,
                    timestamp=self.cur_timestamp,
                    content=model_para))

    def check_and_save(self):
        if self.state == self.total_round_num:
            logger.info('Server: Final evaluation is finished! Starting '
                        'merging results.')
            self.save_best_results()
            self.terminate(msg_type='finish')
            self.state += 1

    def _merge_and_format_eval_results(self):
        formatted_eval_res = self.merge_eval_results_from_all_clients()
        self.history_results = merge_dict_of_results(self.history_results,
                                                     formatted_eval_res)
        self.check_and_save()

    def check_and_move_on(self, min_received_num=None, check_eval_result=False):
        if min_received_num is None:
            min_received_num = self._cfg.federate.sample_client_num
        assert min_received_num <= self.sample_client_num
        move_on_flag = True
        if self.check_buffer(self.state, min_received_num, check_eval_result):
            if not check_eval_result:
                self._perform_federated_aggregation()
                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != self.total_round_num:
                    logger.info(f'-------------- Server: Starting evaluation at the end '
                                f'of round {self.state - 1}. ------------------')
                    self.eval()

                if self.state < self.total_round_num:
                    logger.info(
                        f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()
                    self.msg_buffer['train'][self.state] = dict()
                    # Start a new training round
                    self._start_new_training_round()
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval()
            else:
                self._merge_and_format_eval_results()
                if self.state >= self.total_round_num:
                    self.is_finish = True
        else:
            move_on_flag = False
        return move_on_flag

    def callback_funcs_model_para(self, message):
        logger.info(
            f"server {self.ID} received model_para message "
        )
        if self.is_finish:
            return "finish"

        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content
        self.sampler.change_state(sender, 'idle')
        assert timestamp >= self.cur_timestamp  # for test
        self.cur_timestamp = timestamp

        if round == self.state:
            if round not in self.msg_buffer['train']:
                self.msg_buffer['train'][round] = dict()
            self.msg_buffer['train'][round][sender] = content
            logger.info(f'model_para receive {sender} round: {round} info')
        else:
            logger.info(f'message is out-of-date from {sender} round: {round}')
            self.dropout_num += 1

        move_on_flag = self.check_and_move_on()
        logger.info(f'model_para can move on ? {move_on_flag}')
        return move_on_flag

    def check_client_join_in(self):
        if len(self._cfg.federate.join_in_info) != 0:
            return len(self.join_in_info) == self.client_num
        else:
            return self.join_in_client_num == self.client_num

    def trigger_for_feat_engr(self,
                              trigger_train_func,
                              kwargs_for_trigger_train_func={}):
        """
        Interface for feature engineering, the default operation is none
        """
        trigger_train_func(**kwargs_for_trigger_train_func)

    def trigger_for_start(self):
        if self.check_client_join_in():
            self.trigger_for_feat_engr(
                self.broadcast_model_para, {
                    'msg_type': 'model_para',
                    'sample_client_num': self.sample_client_num
                })
            logger.info(f'----------- Starting training (Round #{self.state}) -------------')

    def callback_funcs_for_join_in(self, message):

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

    def callback_funcs_for_metrics(self, message):
        rnd = message.state
        sender = message.sender
        content = message.content

        if rnd not in self.msg_buffer['eval'].keys():
            self.msg_buffer['eval'][rnd] = dict()

        self.msg_buffer['eval'][rnd][sender] = content

        return self.check_and_move_on(check_eval_result=True)

    def run(self):
        while self.join_in_client_num < self.client_num:
            logger.info('waiting client to join....')
            msg = self.comm_manager.receive()
            self.msg_handlers[msg.msg_type](msg)
        while self.state <= self.total_round_num:
            msg = self.comm_manager.receive()
            move_on_flag = self.msg_handlers[msg.msg_type](msg)
        self.terminate(msg_type='finish')
