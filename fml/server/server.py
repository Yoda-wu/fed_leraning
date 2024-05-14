import fedml
import sys
from ServerManager import FedAvgServerManager

sys.path.append('../..')
sys.path.append('..')


class FedAvgServer:
    """
    FedAvgServer 实现
    参考FedML的fedavg示例，用户需要自己实现服务器端的实体
    """

    def __init__(self, args, device, dataset, model):
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ] = dataset

        self.server_manager = init_server(
            args,
            device,
            args.comm,
            args.rank,
            args.worker_num,
            model,
            train_data_num,
            train_data_global,
            test_data_global,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num_dict,
        )

    def run(self):
        self.server_manager.run()


def init_server(
        args,
        device,
        comm,
        rank,
        client_number,
        model,
        train_data_num,
        train_data_global,
        test_data_global,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
):
    aggregator = get_aggregator(
        args,
        model,
        client_number,
        device,
        train_data_global,
        test_data_global,
        train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict
    )
    backend = args.backend
    server_manager = FedAvgServerManager(args, aggregator, comm, rank, client_number, backend)
    return server_manager


def get_aggregator(
        args,
        model,
        client_number,
        device,
        train_data_global,
        test_data_global,
        train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
):
    from fml.aggregator.aggregator import Aggregator

    return Aggregator(
        args,
        model,
        client_number,
        device,
        train_data_global,
        test_data_global,
        train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
    )


from fedml.data.MNIST.data_loader import download_mnist, load_partition_data_mnist

"""
加载数据集部分，客户端和服务器端都一样
"""


def load_data_cifar(args):
    fedml.logging.info("load_data. dataset_name = %s" % args.dataset)
    centralized = True if (
            args.client_num_in_total == 1 and args.training_type != "cross_silo") else False
    print(f"centralized = {centralized}")
    """
    For shallow NN or linear models, 
    we uniformly sample a fraction of clients each round (as the original FedAvg paper)
    """
    dataset, class_num = fedml.data.load(
        args
    )
    print(f"dataset = {len(dataset)}")
    (train_data_num,
     test_data_num,
     train_data_global,
     test_data_global,
     train_data_local_num_dict,
     train_data_local_dict,
     test_data_local_dict, _
     ) = dataset
    print(f"train_data_local_dict = {train_data_local_dict.keys()} client num  ={class_num}")
    return dataset, class_num


def load_data_mnist(args):
    download_mnist(args.data_cache_dir)
    fedml.logging.info("load_data. dataset_name = %s" % args.dataset)

    """
    Please read through the data loader at to see how to customize the dataset for FedML framework.
    """
    (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = load_partition_data_mnist(
        args,
        args.batch_size,
        train_path=args.data_cache_dir + "/MNIST/train",
        test_path=args.data_cache_dir + "/MNIST/test",
    )
    """
    For shallow NN or linear models, 
    we uniformly sample a fraction of clients each round (as the original FedAvg paper)
    """
    args.client_num_in_total = client_num
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset, class_num


def load_data(args):
    if args.dataset == "mnist":
        return load_data_mnist(args)
    elif args.dataset == "cifar10":
        return load_data_cifar(args)
    else:
        raise Exception("The dataset is not supported")


if __name__ == '__main__':
    args = fedml.init()
    fedml.logging.info(f"client_totoal = {args.client_num_in_total}")
    Device = fedml.device.get_device(args)

    datasets, output_dim = load_data(args)
    from fml.model.lenet5 import LeNet5

    in_channels = 1
    if args.dataset == 'mnist':
        output_dim = 10
    elif args.dataset == 'cifar10':
        output_dim = 10
        in_channels = 3
    model = LeNet5(in_channels=in_channels, num_classes=output_dim)

    server_runner = FedAvgServer(args, Device, datasets, model)

    server_runner.run()
