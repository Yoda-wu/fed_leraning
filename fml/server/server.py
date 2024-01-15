import fedml
# from fedml.cross_silo.server.fedml_server_manager import FedMLServerManager
import sys
from ServerManager import FedAvgServerManager

sys.path.append('../..')
sys.path.append('..')


class FedAvgServer:
    """
    FedAvgServer，作为FedAvg算法里的服务器的角色。
    这个类的主要作用是初始化服务器，然后运行服务器。
    服务器的初始化需要的参数有：
    1. args：命令行参数
    2. device：设备 CPU/GPU
    3. dataset：数据集，这里是MNIST数据集
    4. model：模型，这里是LeNet5

    服务器里主要的方法是run()，这个方法会调用FedMLServerManager的run()方法。

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
    """
    初始化服务器，返回一个FedMLServerManager对象。
    """
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
    """
    返回一个Aggregator对象。
    作为FedAvg算法聚合功能的实现。
    """
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


def load_data(args):
    """
    加载数据集，这里是MNIST数据集。
    """
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


if __name__ == '__main__':
    args = fedml.init()

    Device = fedml.device.get_device(args)

    datasets, output_dim = fedml.data.load(args)
    from fml.model.lenet5 import LeNet5

    Model = LeNet5(num_classes=output_dim)

    server_runner = FedAvgServer(args, Device, datasets, Model)

    server_runner.run()
