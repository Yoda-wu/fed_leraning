# from utils.config_parser import load_yml_conf
# from prettytable import PrettyTable
# yml = load_yml_conf('./conf/fedml_conf.yaml')
# print(yml['common_args']['training_type'])
# config_table = PrettyTable()
# config_table.add_column('Configuration Key', list(yml.keys()))
# config_table.add_column('Configuration Value', list(yml.values()))
# config_table = str(config_table)
# config_list = config_table.split('\n')
# print('----------------------------CONFIGURATION----------------------------')
# for config_entry in config_list:
#     print(config_entry)
# print('----------------------------CONFIGURATION----------------------------')
#
# import fml
# fml.init()

class model:
    def __init__(self):
        self.a = 1

class model_adapter:
    def __init__(self, model):
        self.model = model
m = model()
ma = model_adapter(m)
m_list = [m]
for i in range(10):
    mm = m_list[0]
    mm.a += 1

print(m_list[0].a)
print(ma.model.a)