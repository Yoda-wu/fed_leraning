import fedml
from model.lenet5 import LeNet5
from fedml.data.MNIST.data_loader import download_mnist, load_partition_data_mnist
from fedml.data.cifar10.data_loader import load_partition_data_cifar10
from fedml import FedMLRunner

def load_data(args):
    fedml.logging.info("load_data. dataset_name = %s" % args.dataset)
    centralized = True if (args.client_num_in_total == 1 and args.training_type != "cross_silo") else False
    print(f"centralized = {centralized}")
    # download_mnist(args.data_cache_dir)
    # fedml.logging.info("load_data. dataset_name = %s" % args.dataset)
    #
    # """
    # Please read through the data loader at to see how to customize the dataset for FedML framework.
    # """
    # (
    #     client_num,
    #     train_data_num,
    #     test_data_num,
    #     train_data_global,
    #     test_data_global,
    #     train_data_local_num_dict,
    #     train_data_local_dict,
    #     test_data_local_dict,
    #     class_num,
    # ) = load_partition_data_mnist(
    #     args,
    #     args.batch_size,
    #     train_path=args.data_cache_dir + "/MNIST/train",
    #     test_path=args.data_cache_dir + "/MNIST/test",
    # )
    """
    For shallow NN or linear models, 
    we uniformly sample a fraction of clients each round (as the original FedAvg paper)
    """
    dataset , client_num =fedml.data.load(
        args
    )
    print(f"dataset = {len(dataset)}")
    ( train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,_
      ) = dataset
    print(f"train_data_local_dict = {train_data_local_dict.keys()} client num  ={client_num}")
    args.client_num_in_total = client_num

    return dataset, client_num

if __name__ == "__main__":
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load_data(args)

    # load model (the size of MNIST image is 28 x 28)
    model = LeNet5(output_dim)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()