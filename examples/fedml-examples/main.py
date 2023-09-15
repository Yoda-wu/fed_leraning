import fedml
import time 
from fedavg import FedAvgRunner
if __name__ == '__main__':
    args = fedml.init()

    print(f"client_number = {args.client_num_in_total}")
    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)
    start = time.time()
    print(f" 2 client_number = {args.client_num_in_total}")
    # start training
    fedml_runner = FedAvgRunner(args, device, dataset, model)
    fedml_runner.train()
    end = time.time()
    print("time cost: ", end - start)