import time
import torch
import numpy as np
import torch.nn as nn
import copy
from fedml.core.alg_frame.server_aggregator import ServerAggregator
from fedml.core import Context
import logging
from fedml.ml.engine import ml_engine_adapter
class FedAvgAggregator(ServerAggregator):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.model = model
        self.args = args
        self.id = 0
        self.cpu_transfer = False if not hasattr(self.args, "cpu_transfer") else self.args.cpu_transfer


    def get_model_params(self):
        if self.cpu_transfer:
            return self.model.cpu().state_dict()
        return self.model.state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)
    def aggregate(self, model_list):
        training_num = 0
        for i in range(len(model_list)):
            local_sample_num, local_model_params = model_list[i]
            training_num += local_sample_num
        num0, avg_param = model_list[0]
        for k in avg_param.keys():
            for i in range(0, len(model_list)):
                local_sample_num, local_model_params = model_list[i]
                w = local_sample_num / training_num
                if i == 0:
                    avg_param[k] = local_model_params[k] * w
                else:
                    avg_param[k] += local_model_params[k] * w
        return avg_param

    def _test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            "test_correct": 0,
            "test_loss": 0,
            "test_precision": 0,
            "test_recall": 0,
            "test_total": 0,
        }
        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)  # pylint: disable=E1102
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()
                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                if len(target.size()) == 1:  #
                    metrics["test_total"] += target.size(0)
                elif len(target.size()) == 2:  # for tasks of next word prediction
                    metrics["test_total"] += target.size(0) * target.size(1)
        return metrics

    def test(self, test_data, device, args):
        # test data
        test_num_samples = []
        test_tot_corrects = []
        test_losses = []

        metrics = self._test(test_data, device, args)

        test_tot_correct, test_num_sample, test_loss = (
            metrics["test_correct"],
            metrics["test_total"],
            metrics["test_loss"],
        )
        test_tot_corrects.append(copy.deepcopy(test_tot_correct))
        test_num_samples.append(copy.deepcopy(test_num_sample))
        test_losses.append(copy.deepcopy(test_loss))

        # test on test dataset
        test_acc = sum(test_tot_corrects) / sum(test_num_samples)
        test_loss = sum(test_losses) / sum(test_num_samples)
        stats = {"test_acc": test_acc, "test_loss": test_loss}
        logging.info(stats)
        return test_acc, test_loss, None, None

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
        train_num_samples = []
        train_tot_corrects = []
        train_losses = []
        for client_idx in range(self.args.client_num_in_total):
            # train data
            metrics = self._test(train_data_local_dict[client_idx], device, args)
            train_tot_correct, train_num_sample, train_loss = (
                metrics["test_correct"],
                metrics["test_total"],
                metrics["test_loss"],
            )
            train_tot_corrects.append(copy.deepcopy(train_tot_correct))
            train_num_samples.append(copy.deepcopy(train_num_sample))
            train_losses.append(copy.deepcopy(train_loss))
            # logging.info("testing client_idx = {}".format(client_idx))

        # test on training dataset
        train_acc = sum(train_tot_corrects) / sum(train_num_samples)
        train_loss = sum(train_losses) / sum(train_num_samples)

        stats = {"training_acc": train_acc, "training_loss": train_loss}
        logging.info(stats)

        return True
class Aggregator:

    def __init__(
            self,
            args,
            model,
            client_number,
            device,
            train_data_global,
            test_data_global,
            all_train_data_num,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num_dict,
    ):
        self.args = args
        self.aggregator = FedAvgAggregator(model, args)

        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = test_data_global
        self.all_train_data_num = all_train_data_num

        Context().add(Context.KEY_TEST_DATA, self.val_global)

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.client_num = client_number
        self.device = device
        self.args.device = device
        logging.info("self.device = {}".format(self.device))
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def get_global_model_params(self):
        return self.aggregator.get_model_params()

    def set_global_model_params(self, model_param):
        self.aggregator.set_model_params(model_param)

    def check_whether_all_receive(self):
        logging.debug("client_num = {}".format(self.client_num))
        for idx in range(self.client_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)

        # for dictionary model_params, we let the user level code to control the device
        if type(model_params) is not dict:
            model_params = ml_engine_adapter.model_params_to_device(self.args, model_params, self.device)

        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True


    def aggregate(self):
        start = time.time()
        model_list = []

        for idx in range(self.client_num):
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))

        model_list_index = [i for i in range(len(model_list))]

        Context().add(Context.KEY_CLIENT_MODEL_LIST, model_list)

        averaged_param = self.aggregator.aggregate(model_list)

        self.set_global_model_params(averaged_param)
        end = time.time()
        logging.info(f'aggregate time cost: {end - start}s')
        return  averaged_param, model_list, model_list_index

    def data_silo_selection(self, round_idx, client_num_in_total, client_num_per_round):
        logging.info(
            "client_num_in_total = %d, client_num_per_round = %d" % (client_num_in_total, client_num_per_round)
        )
        assert client_num_in_total >= client_num_per_round

        if client_num_in_total == client_num_per_round:
            return [i for i in range(client_num_per_round)]
        else:
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            data_silo_index_list = np.random.choice(range(client_num_in_total), client_num_per_round, replace=False)
            return data_silo_index_list
    def client_selection(self, round_idx, client_id_list_in_total, client_num_per_round):

        if client_num_per_round == len(client_id_list_in_total):
            return client_id_list_in_total
        np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        client_id_list_in_this_round = np.random.choice(client_id_list_in_total, client_num_per_round, replace=False)
        logging.info(f"client_selection： {client_id_list_in_this_round}")
        return client_id_list_in_this_round

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def test_on_server(self, round_idx):
        if round_idx == self.args.comm_round - 1 :
            logging.info(f"============== test on server for all client {round_idx} ==============")
            self.aggregator.test_all(
                self.train_data_local_dict,
                self.test_data_local_dict,
                self.device,
                self.args
            )
            metrics = self.aggregator.test(self.test_global, self.device, self.args)
            logging.info(f"test on server metrics is {metrics}")
