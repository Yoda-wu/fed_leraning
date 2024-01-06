import time
import torch
import torch.nn as nn
from fedml.ml.engine import ml_engine_adapter
from fedml.ml.trainer.trainer_creator import create_model_trainer
import logging
from fedml.core.alg_frame.client_trainer import ClientTrainer


class FedAvgModelTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate)
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )

    def train_iterations(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []

        current_steps = 0
        current_epoch = 0
        while current_steps < args.local_iterations:
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                labels = labels.long()
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                current_steps += 1
                if current_steps == args.local_iterations:
                    break
            current_epoch += 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, current_epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                target = target.long()
                loss = criterion(pred, target)  # pylint: disable=E1102

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        return metrics


class Trainer:
    def __init__(self,
                 args,
                 device,
                 client_rank,
                 model,
                 train_data_num,
                 train_data_local_num_dict,
                 train_data_local_dict,
                 test_data_local_dict,
                 ):
        self.test_local = None
        self.local_sample_number = None
        self.train_local = None
        ml_engine_adapter.model_to_device(args, model, device)
        self.model_trainer = FedAvgModelTrainer(model, args)
        # client rank 从1 开始
        client_index = client_rank - 1
        self.model_trainer.set_id(client_index)
        self.client_index = client_index
        self.client_rank = client_rank
        self.device = device
        self.args = args
        self.train_data_num = train_data_num
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

    def set_epoch(self, epoch):
        self.args.epochs = epoch

    def train(self, round_idx):
        self.args.round_idx = round_idx

        tick = time.time()
        self.model_trainer.train(self.train_local, self.device, self.args)
        logging.info(f'[client-{self.client_rank}] finish train, cost time: {time.time() - tick} s')
        weights = self.model_trainer.get_model_params()
        return weights, self.local_sample_number

    def test(self, round_idx):
        self.args.round_idx = round_idx
        metrics = self.model_trainer.test(self.test_local, self.device, self.args)
        logging.info(
            f"[client-{self.client_rank}] finish test, loss : {metrics['test_loss']} and acc : {float(metrics['test_correct']) / float(metrics['test_total'])}")
        return metrics

    def update_model(self, model_params):
        self.model_trainer.set_model_params(model_params)

    def update_dataset(self, client_index=None):
        self.client_index = client_index

        if self.train_data_local_dict is not None:
            self.train_local = self.train_data_local_dict[client_index]
        else:
            self.train_local = None

        if self.train_data_local_num_dict is not None:
            self.local_sample_number = self.train_data_local_num_dict[client_index]
        else:
            self.local_sample_number = 0

        if self.test_data_local_dict is not None:
            self.test_local = self.test_data_local_dict[client_index]
        else:
            self.test_local = None

        self.model_trainer.update_dataset(self.train_local, self.test_local, self.local_sample_number)
