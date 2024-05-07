import json
import platform
import time

from fedml.core.distributed.fedml_comm_manager import FedMLCommManager
from fedml.utils.logging import logger
from fedml.core.distributed.communication.message import Message
from fml.message_define import MyMessage

"""
FedML提供了两种ClientManager，分别是ClientMasterManager和ClientSlaveManager
其中只有前者是可以与服务器直接通信，后者只能与Master节点通信。
这里不选择直接继承ClientMasterManager，而是直接继承FedMLCommManager，实现自己的ClientManager的原因如下：
1. Trainer的包装——ClientMasterManager中的Trainer是通过TrainerDistAdapter来包装的，而TrainerDistAdapter中的又包装了一层FedMLTrainer，而FedMLTrainer中的又包装了一层ClientTrainer
2. 这里需要自定义与服务器通信的消息处理。
"""
class FedAvgClientManager(FedMLCommManager):
    ONLINE_STATUS_FLAG = "ONLINE"
    RUN_FINISHED_STATUS_FLAG = "FINISHED"

    def __init__(self, args, trainer, comm, client_rank, client_number, backend):
        super().__init__(args, comm, client_rank, size=client_number, backend=backend)
        self.trainer = trainer
        self.args = args
        self.hello = "hello"
        self.num_round = args.comm_round
        self.round_idx = 0
        self.rank = client_rank
        self.client_real_ids = json.loads(args.client_id_list)
        logger.info(f"client_real_ids = {self.client_real_ids}")
        self.client_real_id = self.client_real_ids[0]

        self.has_sent_online_msg = False
        self.is_inited = False

    def is_main_process(self):
        """
        判断是否是主进程
        """
        return getattr(self.trainer, "trainer", None) is None or \
            getattr(self.trainer.trainer, "trainer", None) is None or \
            self.trainer.model_trainer.is_main_process()

    def register_message_receive_handlers(self) -> None:
        """
        注册消息处理函数
        """
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_CONNECTION_IS_READY, self.handle_message_connection_ready
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_CHECK_CLIENT_STATUS, self.handle_message_check_status
        )

        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init)
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.handle_message_receive_model_from_server,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_FINISH, self.handle_message_finish,
        )

    def send_client_status(self, receive_id, status=ONLINE_STATUS_FLAG):
        """
        发送客户端状态
        """
        if self.is_main_process():
            logger.info("send_client_status")
            logger.info(f"self.client_read_id{self.client_real_id}")
            message = Message(MyMessage.MSG_TYPE_C2S_CLIENT_STATUS, self.client_real_id, receive_id)
            sys_name = platform.system()
            message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_STATUS, status)
            message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, sys_name)
            self.send_message(message)

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        """
        发送模型到服务器
        """
        if self.is_main_process():
            tick = time.time()
            message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.client_real_id, receive_id)
            message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
            message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
            self.send_message(message)

            logger.info(f"communication cost time :{time.time() - tick} s")

    def handle_message_connection_ready(self, msg_param):
        """
        处理连接准备就绪消息
        """
        if not self.has_sent_online_msg:
            self.has_sent_online_msg = True
            self.send_client_status(0)

    def handle_message_check_status(self, msg_param):
        """
        处理检查状态消息
        """
        self.send_client_status(0)

    def handle_message_init(self, msg_param):
        """
        处理初始化消息
        """
        if self.is_inited:
            return

        self.is_inited = True

        global_model_param = msg_param.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        data_silo_index = msg_param.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        logger.info(f"data_silo_index = {data_silo_index}")
        self.trainer.update_dataset(int(data_silo_index))
        self.trainer.update_model(global_model_param)
        self.round_idx = 0
        self.train()
        self.test()
        self.round_idx += 1

    def handle_message_receive_model_from_server(self, msg_param):
        """

        处理从服务器接收模型消息 —— 需要额外接收epoch数据
        """
        logger.info("handle_message_receive_model_from_server")
        model_params = msg_param.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_param.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        epoch = msg_param.get(MyMessage.MSG_ARG_KEY_CLIENT_EPOCH)
        self.trainer.set_epoch(int(epoch))
        self.trainer.update_dataset(int(client_index))
        self.trainer.update_model(model_params)
        if self.round_idx < self.num_round:
            self.train()
            self.test()
            self.round_idx += 1
        else:
            self.send_client_status(0, FedAvgClientManager.RUN_FINISHED_STATUS_FLAG)
            self.finish()

    def handle_message_finish(self, msg_param):
        """
        处理结束消息
        """
        logger.info(" ==================finish================== ")
        self.cleanup()

    def cleanup(self):
        """
        结束
        """
        self.send_client_status(0, FedAvgClientManager.RUN_FINISHED_STATUS_FLAG)
        self.finish()

    def train(self):
        """
        本地训练
        """
        logger.info(f"=======training=======  round_id = {self.round_idx}")
        weights, local_sample_num = self.trainer.train(self.round_idx)
        self.send_model_to_server(0, weights, local_sample_num)

    def test(self):
        self.trainer.test(self.round_idx)
    
    def run(self):
        """
        启动
        """
        super().run()