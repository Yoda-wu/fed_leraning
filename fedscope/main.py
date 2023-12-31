import logging
import os
import sys


DEV_MODE = False  # simplify the federatedscope re-setup everytime we change
# the source codes of federatedscope
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger, logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, \
    get_server_cls
from federatedscope.core.configs.config import global_cfg, CfgNode
from federatedscope.core.auxiliaries.runner_builder import get_runner
from client.fedavg_client import FedAvgClient
from server.fedavg_server import FedAvgServer
# 导入目的为了register可以注册到federatedScope框架里
from model import lenet5


if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']


if __name__ == '__main__':
    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # load clients' cfg file
    if args.client_cfg_file:
        client_cfgs = CfgNode.load_cfg(open(args.client_cfg_file, 'r'))
        # client_cfgs.set_new_allowed(True)
        client_cfgs.merge_from_list(client_cfg_opt)
    else:
        client_cfgs = None

    # federated dataset might change the number of clients
    # thus, we allow the creation procedure of dataset to modify the global
    # cfg object
    from dataset.mnist import load_mnist
    data, modified_cfg = load_mnist(config=init_cfg.clone(),
                                  client_cfgs=client_cfgs)
    init_cfg.merge_from_other_cfg(modified_cfg)



    init_cfg.freeze()

    logger.info(f"len( {len(data)} ) ,  keys : {data.keys()}")
    runner = get_runner(data=data,
                        server_class=FedAvgServer,
                        client_class=FedAvgClient,
                        config=init_cfg.clone(),
                        client_configs=client_cfgs)
    _ = runner.run()
