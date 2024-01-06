import fedml

from ClientManager import FedAvgClientManager
from fedml.cross_silo.client.fedml_client_master_manager import ClientMasterManager

class FedAvgClient:
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
            args.client_number,
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
    client_manager: ClientMasterManager = None
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
    client_manager = ClientMasterManager(args, trainer, comm, client_rank, client_number, backend)
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


if __name__ == '__main__':
    args = fedml.init()

    device = fedml.device.get_device(args)

    datasets, output_dim = fedml.data.load(args)
    from fml.model.lenet5 import LeNet5

    model = LeNet5(num_classes=output_dim)

    client_runner = FedAvgClient(args, device, datasets, model)

    client_runner.run()