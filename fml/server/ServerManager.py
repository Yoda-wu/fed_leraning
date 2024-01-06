import logging
import time
import sys

from fedml.core.distributed.communication.message import Message

sys.path.append('../..')
sys.path.append('..')
from fedml.cross_silo.server.fedml_server_manager import FedMLServerManager
from fml.message_define import MyMessage
from fedml.utils.logging import logger
from fedml.core import Context


class FedAvgServerManager(FedMLServerManager):
    def __init__(self, args, aggregator, comm=None, client_rank=0, client_num=0, backend="MQTT_S3"):
        super().__init__(args, aggregator, comm=comm, client_rank=client_rank, client_num=client_num, backend=backend)
        self.client_round_map = {}
        for client_id in self.client_real_ids:
            self.client_round_map[client_id] = self.args.round_idx

    def send_init_msg(self):
        global_model_params = self.aggregator.get_global_model_params()

        global_model_url = None
        global_model_key = None

        client_idx_in_this_round = 0
        for client_id in self.client_id_list_in_this_round:

            if type(global_model_params) is dict:
                client_index = self.data_silo_index_list[client_idx_in_this_round]
                global_model_url, global_model_key = self.send_message_init_config(
                    client_id, global_model_params[client_index], client_index,
                    None, None, client_epoch=2
                )
            else:
                global_model_url, global_model_key = self.send_message_init_config(
                    client_id, global_model_params, self.data_silo_index_list[client_idx_in_this_round],
                    global_model_url, global_model_key
                )
            client_idx_in_this_round += 1

    def send_message_init_config(self, receive_id, global_model_params, datasilo_index,
                                 global_model_url=None, global_model_key=None, client_epoch=0):
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
            logging.info("self.client_id_list_in_this_round = {}".format(self.client_id_list_in_this_round))
            new_client_id_list_in_this_round = []
            for client_idx in model_list_indexes:
                new_client_id_list_in_this_round.append(self.client_id_list_in_this_round[client_idx])
            logging.info("new_client_id_list_in_this_round = {}".format(new_client_id_list_in_this_round))
            Context().add(Context.KEY_CLIENT_ID_LIST_IN_THIS_ROUND, new_client_id_list_in_this_round)
            logging.info(f"AggregationTime: {time.time() - tick}, round: {self.args.round_idx}")

            self.aggregator.test_on_server(self.args.round_idx)

            self.client_id_list_in_this_round = self.aggregator.client_selection(
                self.args.round_idx, self.client_real_ids, self.args.client_num_per_round
            )

            self.data_silo_index_list = self.aggregator.data_silo_selection(
                self.args.round_idx, self.args.client_num_in_total, len(self.client_id_list_in_this_round),
            )
            Context().add(Context.KEY_CLIENT_ID_LIST_IN_THIS_ROUND, self.client_id_list_in_this_round)

            client_idx_in_this_round = 0
            global_model_url = None
            global_model_key = None
            for receiver_id in self.client_id_list_in_this_round:
                self.client_round_map[receiver_id] += 1
                client_epoch = 2
                if self.client_round_map[receiver_id] >= 3 :
                    client_epoch = 1
                client_index = self.data_silo_index_list[client_idx_in_this_round]

                if type(global_model_params) is dict:
                    # compatible with the old version that, user did not give {-1 : global_params_dict}
                    global_model_url, global_model_key = self.send_message_diff_sync_model_to_client(
                        receiver_id, global_model_params[client_index], client_index, client_epoch=client_epoch
                    )
                else:
                    global_model_url, global_model_key = self.send_message_sync_model_to_client(
                        receiver_id, global_model_params, client_index, global_model_url, global_model_key,
                        client_epoch=client_epoch
                    )
                client_idx_in_this_round += 1

            # if user give {-1 : global_params_dict}, then record global_model url separately
            # Note MPI backend does not have rank -1
            if self.backend != "MPI" and type(global_model_params) is dict and (-1 in global_model_params.keys()):
                global_model_url, global_model_key = self.send_message_diff_sync_model_to_client(
                    -1, global_model_params[-1], -1
                )

            self.args.round_idx += 1

            logging.info("\n\n==========end {}-th round training===========\n".format(self.args.round_idx))

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index,
                                          global_model_url=None, global_model_key=None, client_epoch=0):
        if self.is_main_process():
            tick = time.time()
            logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
            message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id, )
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

    def send_message_diff_sync_model_to_client(self, receive_id, client_model_params, client_index, client_epoch=0):
        global_model_url = None
        global_model_key = None

        if self.is_main_process():
            tick = time.time()
            logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
            message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id, )
            message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, client_model_params)
            message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
            message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, "PythonClient")
            message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_EPOCH, str(client_epoch))
            self.send_message(message)

            logging.info({"Communiaction/Send_Total": time.time() - tick})

            global_model_url = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL)
            global_model_key = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_KEY)

        return global_model_url, global_model_key
