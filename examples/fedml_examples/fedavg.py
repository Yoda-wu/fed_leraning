import copy
import random
import logging 
import fedml
import numpy as np
import torch
import wandb
import fedml.cross_silo.client.client_initializer as ClientInitializer
import fedml.cross_silo.server.server_initializer as ServerInitializer
from fedml.ml.aggregator.aggregator_creator import create_server_aggregator
from fedml import mlops


class Client:
    def __init__(
        self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, trainer,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

        self.args = args
        self.device = device
        self.trainer = trainer



class FedAvgRunner(object):


    def __init__(self, args, device, dataset, model) -> None:
        self.device = device
        self.args = args
       
        print(f"client_number = {args.client_num_in_total}")
        self.args.scenario = None
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
        self.max_acc = 0
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        
        self.model = model
        # logging.info("self.model_trainer = {}".format(self.model_trainer))
        server_aggregator = create_server_aggregator(model, args)
        server_aggregator.set_id(0)
        worker_num = self.args.client_num_per_round
        self.aggregator = ServerInitializer.FedMLAggregator(
            train_data_global,
            test_data_global,
            train_data_num,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num_dict,
            worker_num,
            device,
            args,
            server_aggregator,
        )
        self._setup_clients(
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict,
        )
    def _setup_clients(
        self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict,
    ):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(1, self.args.client_num_per_round + 1):
            model_trainer = ClientInitializer.get_trainer_dist_adapter(
                self.args,
                self.device,
                client_idx,
                self.model,
                self.train_data_num_in_total,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                None,
            )
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                model_trainer,
            )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        # logging.info("self.model_trainer = {}".format(self.model_trainer))
        w_global = self.aggregator.get_global_model_params()
        mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        mlops.log_round_info(self.args.comm_round, -1)
        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self.aggregator.client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            logging.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                logging.info("idx = " + str(idx))
                logging.info("idx = " + str(len(self.client_list)))
                # update dataset
                client_idx = client_indexes[idx]
                client.trainer.update_dataset(client_idx)
                
                # train on new dataset
                mlops.event("train", event_started=True, event_value="{}_{}".format(str(round_idx), str(idx)))
                weights, local_sample_num = client.trainer.train(round_idx)
                # w = client.train(copy.deepcopy(w_global))
                mlops.event("train", event_started=False, event_value="{}_{}".format(str(round_idx), str(idx)))
                # self.logging.info("local weights = " + str(w))
                print(f"before aggregator add local trainer result client_idx = {client_idx}")
                self.aggregator.add_local_trained_result(
                    client_idx, weights, local_sample_num
                )
                # w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            # update global weights
            mlops.event("agg", event_started=True, event_value=str(round_idx))
            # w_global = self._aggregate(w_locals)
            veraged_params, model_list, model_list_idxes = self.aggregator.aggregate()

            # self.model_trainer.set_model_params(w_global)
            mlops.event("agg", event_started=False, event_value=str(round_idx))
            self.aggregator.test_on_server_for_all_clients(self.args.round_idx)
            # test results
            # at last round
            # if round_idx == self.args.comm_round - 1:
            #     self._local_test_on_all_clients(round_idx)
            # # per {frequency_of_the_test} round
            # elif round_idx % self.args.frequency_of_the_test == 0:
            #     if self.args.dataset.startswith("stackoverflow"):
            #         self._local_test_on_validation_set(round_idx)
            #     else:
            #         self._local_test_on_all_clients(round_idx)

            mlops.log_round_info(self.args.comm_round, round_idx)

        mlops.log_training_finished_status()
        mlops.log_aggregation_finished_status()

    # def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
    #     if client_num_in_total == client_num_per_round:
    #         client_indexes = [client_index for client_index in range(client_num_in_total)]
    #     else:
    #         num_clients = min(client_num_per_round, client_num_in_total)
    #         np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
    #         client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
    #     logging.info("client_indexes = %s" % str(client_indexes))
    #     return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    # def _aggregate(self, w_locals):
    #     training_num = 0
    #     for idx in range(len(w_locals)):
    #         (sample_num, averaged_params) = w_locals[idx]
    #         training_num += sample_num

    #     (sample_num, averaged_params) = w_locals[0]
    #     for k in averaged_params.keys():
    #         for i in range(0, len(w_locals)):
    #             local_sample_number, local_model_params = w_locals[i]
    #             w = local_sample_number / training_num
    #             if i == 0:
    #                 averaged_params[k] = local_model_params[k] * w
    #             else:
    #                 averaged_params[k] += local_model_params[k] * w
    #     print(f"averaged_params key  {averaged_params}")
    #     return averaged_params

    # def _aggregate_noniid_avg(self, w_locals):
    #     """
    #     The old aggregate method will impact the model performance when it comes to Non-IID setting
    #     Args:
    #         w_locals:
    #     Returns:
    #     """
    #     (_, averaged_params) = w_locals[0]
    #     for k in averaged_params.keys():
    #         temp_w = []
    #         for (_, local_w) in w_locals:
    #             temp_w.append(local_w[k])
    #         averaged_params[k] = sum(temp_w) / len(temp_w)
    #     return averaged_params

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(
                0,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
            )
            # train data
            train_local_metrics = client.local_test(False)
            train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
            train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
            train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
            test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
            test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))

        # test on training dataset
        train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
        train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

        # test on test dataset
        test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
        test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

        stats = {"training_acc": train_acc, "training_loss": train_loss}
        if self.args.enable_wandb:
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})

        mlops.log({"Train/Acc": train_acc, "round": round_idx})
        mlops.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)
        self.max_acc = max(self.max_acc, train_acc)
        stats = {"test_acc": test_acc, "test_loss": test_loss, "round_id": round_idx, "max_acc" : self.max_acc}

        if self.args.enable_wandb:
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})

        mlops.log({"Test/Acc": test_acc, "round": round_idx})
        mlops.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)

    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {"test_acc": test_acc, "test_loss": test_loss}
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})

            mlops.log({"Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Test/Loss": test_loss, "round": round_idx})

        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_pre = test_metrics["test_precision"] / test_metrics["test_total"]
            test_rec = test_metrics["test_recall"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {
                "test_acc": test_acc,
                "test_pre": test_pre,
                "test_rec": test_rec,
                "test_loss": test_loss,
            }
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Pre": test_pre, "round": round_idx})
                wandb.log({"Test/Rec": test_rec, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})

            mlops.log({"Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Test/Pre": test_pre, "round": round_idx})
            mlops.log({"Test/Rec": test_rec, "round": round_idx})
            mlops.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)
