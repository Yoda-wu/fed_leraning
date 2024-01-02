from flwr.server import ServerConfig
from prettytable import PrettyTable
class Configuration:

    def __init__(self, args):
        self.algorithm = args.algorithm
        self.client_number = args.client_number
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.model = args.model
        self.frac_clients = args.frac_clients
        self.device = args.device
        self.available_clients = args.available_clients
        self.server_config = ServerConfig(num_rounds=args.num_rounds)

        self.config_dict = {}

    def generate_config_dict(self):
        '''
        记录配置表，以便后续分析数据读取
        '''
        self.config_dict['algorithm'] = self.algorithm
        self.config_dict['model'] = self.model
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


