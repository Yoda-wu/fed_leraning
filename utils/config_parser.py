import argparse
import yaml


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()

    def set_arguments(self):
        # GPU ID
        self.parser.add_argument('--gpu', type=str, help='gpu ids to use e.g. 0,1,2,...')
        # 联邦学习算法
        self.parser.add_argument('--algorithm', type=str,
                                 help='distributed semi-supervised learning algorithm i.e. my_dssl, gossip_fixmatch, gossip, gossip_uy_known_fixmatch')
        # 客户端数目
        self.parser.add_argument('--client-number', type=str,
                                 help='client number i.e. total client')
        # 运行轮次
        self.parser.add_argument('--num-rounds', type=str, help='total training round')
        # 数据集选择
        self.parser.add_argument('--dataset', type=str,
                                 help='dataset i.e. mnist, fmnist (fashion mnist) or cifar10')
        # 模型选择
        self.parser.add_argument('--model', type=str, help='model i.e. lenet5, resnet')
        # 客户端选择算法
        self.parser.add_argument('--client-selection', type=str,
                                 help='client selection algorithm i.e. random')
        # 每轮参与训练的客户端比例
        self.parser.add_argument('--frac-clients', type=float, help='fraction of clients per round')
        # 每轮客户端数量
        self.parser.add_argument('--available-clients', type=float, help='clients per round')
        # batch_size
        self.parser.add_argument('--batch-size', type=str,
                                 help='initial batch size i.e. 16, 32, 64')
        self.parser.add_argument('--server-address', type=str, help='server address ip:port')
        self.parser.add_argument("--node-id", type=int,
                                 help='node id which is associate to client-number i.e client-number = 10 then node-id in [0,1,...,9]')

    def parse(self):
        args, unparsed = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args


def load_yml_conf(yaml_file):
    with open(yaml_file) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data
