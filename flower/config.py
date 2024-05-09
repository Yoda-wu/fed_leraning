from flwr.server import ServerConfig
from prettytable import PrettyTable


class Configuration:
    """
    这里是我自己写的配置类，方便使用脚本启动。这里不是Flower的一部分。
    """

    def __init__(self, args):
        self.algorithm = args.algorithm
        self.client_number = int(args.client_number)
        self.dataset = args.dataset
        self.batch_size = int(args.batch_size)
        self.model = None

        self.frac_clients = args.frac_clients
        self.device = 'cpu'
        self.available_clients = int(args.available_clients)
        self.server_config = ServerConfig(num_rounds=int(args.num_rounds))
        self.config_dict = {}
        self.num_classes = 10
        self.server_address = args.server_address
        if self.dataset in ['mnist', 'cifar10']:
            self.num_classes = 10
        if self.dataset == 'mnist':
            self.in_channels = 1
        if self.dataset == 'cifar10':
            self.in_channels = 3
        if args.model == 'lenet5':
            from flower.model.lenet5 import lenet5
            self.model = lenet5(in_channel=self.in_channels, num_classes=self.num_classes)

    def generate_config_dict(self):
        '''
        记录配置表，以便后续分析数据读取
        '''
        self.config_dict['algorithm'] = self.algorithm
        self.config_dict['model'] = self.model.__class__.__name__
        self.config_dict['dataset'] = self.dataset
        self.config_dict['batch_size'] = self.batch_size
        self.config_dict['frac_clients'] = self.frac_clients
        self.config_dict['available_clients'] = self.available_clients
        self.config_dict['num_rounds'] = self.server_config.num_rounds

    def get_config_dict(self):
        return self.config_dict

    def show_configuration(self):
        config_table = PrettyTable()
        config_table.add_column('Configuration Key', list(self.config_dict.keys()))
        config_table.add_column('Configuration Value', list(self.config_dict.values()))
        config_table = str(config_table)
        config_list = config_table.split('\n')
        print('----------------------------CONFIGURATION----------------------------')
        for config_entry in config_list:
            print(config_entry)
        print('----------------------------CONFIGURATION----------------------------')
