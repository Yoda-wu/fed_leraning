import logging
import time
import sys
import fedml
from fedml.core.distributed.communication.message import Message

sys.path.append('../..')
sys.path.append('..')
from fedml.cross_silo.server.fedml_server_manager import FedMLServerManager
from fml.message_define import MyMessage
from fedml.core import Context

"""
与Client端不同，FedML提供一个比较通用的Server端实现，因为Server的行为比较固定，并且在FedML没有对其进行过多的封装，用户可以根据自己的需求进行定制
"""


class FedAvgServerManager(FedMLServerManager):
    """
    FedAvgServerManager，作为FedAvg算法里的服务器主要功能实现的角色类。
    这个类继承FedML提供的FedMLServerManager类，在这个框架下定义FedAvg的行为。

    这个类主要负责与客户端的通信，包括：
    1. 初始化客户端
    2. 接收客户端的模型
    3. 发送模型给客户端
    4. 客户端选择
    等
    """

    def __init__(self, args, aggregator, comm=None, client_rank=0, client_num=0, backend="MQTT_S3"):
        super().__init__(args, aggregator, comm=comm, client_rank=client_rank,
                         client_num=client_num, backend=backend)
        self.begin_timer = None
        self.client_round_map = {}
        fedml.logging.info(f"client_num = {args.client_num_in_total}")
        for client_id in self.client_real_ids:
            self.client_round_map[client_id] = self.args.round_idx

    def handle_message_connection_ready(self, msg_params):
        """
        处理来自Client的消息，主要是检查Client的状态。
        这里重写的原因是，FedML的ServerManager中使用到了data_silo_index_list。感觉是冗余的。因此重写这个方法
        """
        if not self.is_initialized:
            self.client_id_list_in_this_round = self.aggregator.client_selection(
                self.args.round_idx, self.client_real_ids, self.args.client_num_per_round
            )
            # check client status in case that some clients start earlier than the server
            client_idx_in_this_round = 0
            for client_id in self.client_id_list_in_this_round:
                try:
                    self.send_message_check_client_status(
                        client_id, client_id,
                    )
                    logging.info("Connection ready for client" + str(client_id))
                except Exception as e:
                    logging.info("Connection not ready for client" + str(client_id))
                client_idx_in_this_round += 1

    def send_init_msg(self):
        """
        向Client发送初始化消息，调用send_message_init_config方法
        """
        self.begin_timer = time.time()
        global_model_params = self.aggregator.get_global_model_params()
        global_model_url = None
        global_model_key = None
        client_idx_in_this_round = 0
        logging.info(
            f"the type of global_model_param is {type(global_model_params)} and"
            f" {type(global_model_params) is dict} and client_id_list_in_this_round = {self.client_id_list_in_this_round}")

        for client_id in self.client_id_list_in_this_round:
            global_model_url, global_model_key = self.send_message_init_config(
                client_id, global_model_params,
                client_id,
                global_model_url, global_model_key
            )
            client_idx_in_this_round += 1

    def send_message_init_config(self, receive_id, global_model_params, datasilo_index,
                                 global_model_url=None, global_model_key=None, client_epoch=0):
        """
        向Client发送初始化消息
        """
        if self.is_main_process():
            tick = time.time()
            message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
            if global_model_url is not None:
                message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL, global_model_url)
            if global_model_key is not None:
                message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_KEY, global_model_key)
            message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
            message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(datasilo_index))
            message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, "PythonClient")
            self.send_message(message)
            global_model_url = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL)
            global_model_key = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_KEY)
            logging.info({"Communiaction/Send_Total": time.time() - tick})
        return global_model_url, global_model_key

    def handle_message_receive_model_from_client(self, msg_params):
        """
        处理来自Client的消息，主要的聚合逻辑在这里
        """
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.aggregator.add_local_trained_result(
            self.client_real_ids.index(sender_id), model_params, local_sample_number
        )
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info(f"all_receiver : {b_all_received}")
        if b_all_received:
            tick = time.time()
            global_model_params, model_list, model_list_indexes = self.aggregator.aggregate()
            logging.info(
                "self.client_id_list_in_this_round = {}".format(self.client_id_list_in_this_round))
            new_client_id_list_in_this_round = []
            for client_idx in model_list_indexes:
                new_client_id_list_in_this_round.append(
                    self.client_id_list_in_this_round[client_idx])
            logging.info(
                "new_client_id_list_in_this_round = {}".format(new_client_id_list_in_this_round))
            Context().add(Context.KEY_CLIENT_ID_LIST_IN_THIS_ROUND,
                          new_client_id_list_in_this_round)
            logging.info(f"AggregationTime: {time.time() - tick}, round: {self.args.round_idx}")

            self.aggregator.test_on_server(self.args.round_idx)
            logging.info(f"client_id list {self.client_real_ids}")
            self.client_id_list_in_this_round = self.aggregator.client_selection(
                self.args.round_idx, self.client_real_ids, self.args.client_num_per_round
            )

            Context().add(Context.KEY_CLIENT_ID_LIST_IN_THIS_ROUND,
                          self.client_id_list_in_this_round)

            client_idx_in_this_round = 0
            global_model_url = None
            global_model_key = None
            logging.info(f"the type of global_model_param is {type(global_model_params)}")
            for receiver_id in self.client_id_list_in_this_round:
                self.client_round_map[receiver_id] += 1
                client_epoch = 2
                if self.client_round_map[receiver_id] >= 3:
                    client_epoch = 1
                global_model_url, global_model_key = self.send_message_sync_model_to_client(
                    receiver_id, global_model_params, receiver_id, global_model_url,
                    global_model_key,
                    client_epoch=client_epoch
                )
                client_idx_in_this_round += 1
            self.args.round_idx += 1

            logging.info(
                "\n\n==========end {}-th round training===========\n".format(self.args.round_idx))

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index,
                                          global_model_url=None, global_model_key=None,
                                          client_epoch=0):
        """
        向Client发送模型同步消息
        """
        if self.is_main_process():
            tick = time.time()
            logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
            message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(),
                              receive_id, )
            message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
            if global_model_url is not None:
                message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL, global_model_url)
            if global_model_key is not None:
                message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_KEY, global_model_key)
            message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
            message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, "PythonClient")
            message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_EPOCH, str(client_epoch))
            self.send_message(message)

            logging.info({"Communiaction/Send_Total": time.time() - tick})

            global_model_url = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL)
            global_model_key = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_KEY)

        return global_model_url, global_model_key
