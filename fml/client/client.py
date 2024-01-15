import fedml
import sys

sys.path.append('../..')
sys.path.append('..')
from ClientManager import FedAvgClientManager
from fedml.cross_silo.client.fedml_client_master_manager import ClientMasterManager


class FedAvgClient:
    """
    FedAvgClient，作为FedAvg算法里的客户端的角色。
    这个类的主要作用是初始化客户端，然后运行客户端。
    客户端的初始化需要的参数有：
    1. args：命令行参数
    2. device：设备 CPU/GPU
    3. dataset：数据集，这里是MNIST数据集
    4. model：模型，这里是LeNet5

    客户端里主要的方法是run()，这个方法会调用FedMLClientManager的run()方法。
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
        self.client_manager = init_client(
            args,
            device,
            args.comm,
            args.rank,
            args.worker_num,
            model,
            train_data_num,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict
        )

    def run(self):
        self.client_manager.run()


def init_client(
        args,
        device,
        comm,
        client_rank,
        client_number,
        model,
        train_data_num,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict
):
    """
    初始化客户端的方法
    这里会初始化本地训练器trainer
    以及客户端管理器client_manager
    """
    client_manager: FedAvgClientManager = None
    backend = args.backend
    trainer = get_trainer(
        args,
        device,
        client_rank,
        model,
        train_data_num,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
    )
    client_manager = FedAvgClientManager(args, trainer, comm, client_rank, client_number, backend)
    return client_manager


def get_trainer(
        args,
        device,
        client_rank,
        model,
        train_data_num,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,

):
    """
    初始化本地训练器trainer
    """
    from fml.trainer.trainer import Trainer
    return Trainer(
        args,
        device,
        client_rank,
        model,
        train_data_num,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
    )


from fedml.data.MNIST.data_loader import download_mnist, load_partition_data_mnist


def load_data(args):
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

    device = fedml.device.get_device(args)

    datasets, output_dim = load_data(args)
    from fml.model.lenet5 import LeNet5

    model = LeNet5(num_classes=output_dim)

    client_runner = FedAvgClient(args, device, datasets, model)

    client_runner.run()
