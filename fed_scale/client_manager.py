from typing import List
import numpy as np

from fedscale.cloud.client_manager import ClientManager


class FedAvgClientManager(ClientManager):
    """
    执行客户端选择功能
    """

    def __init__(self, mode, args, sample_seed=233):
        super(FedAvgClientManager, self).__init__(mode, args, sample_seed)

    def select_participants(self, num_of_clients: int, cur_time: float = 0) -> List[int]:
        clients_available = self.feasibleClients
        if len(clients_available) < num_of_clients:
            return clients_available
        return np.random.choice(clients_available, num_of_clients, replace=False).tolist()
