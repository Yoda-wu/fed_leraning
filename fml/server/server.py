import fedml
from fedml.cross_silo.server.fedml_server_manager import FedMLServerManager

class FedAvgServer:
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
    server_manager = FedMLServerManager(args, aggregator, comm, rank, client_number, backend)
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


if __name__ == '__main__':
    args = fedml.init()

    device = fedml.device.get_device(args)

    datasets, output_dim = fedml.data.load(args)
    from fml.model.lenet5 import LeNet5
    model = LeNet5(num_classes=output_dim)

    server_runner = FedAvgServer(args, device, datasets, model)

    server_runner.run()
