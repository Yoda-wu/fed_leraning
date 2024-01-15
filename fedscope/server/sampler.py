import numpy as np
from federatedscope.core.sampler import Sampler


class RandomSampler(Sampler):
    """
    客户端随机选择算法实现
    """
    def __init__(self, client_num):
        super(RandomSampler, self).__init__(client_num)

    def sample(self, size):
        """
        To sample clients
        """
        idle_clients = np.nonzero(self.client_state)[0]
        # print(idle_clients)
        sampled_clients = np.random.choice(idle_clients,
                                           size=size,
                                           replace=False).tolist()
        self.change_state(sampled_clients, 'working')
        return sampled_clients
