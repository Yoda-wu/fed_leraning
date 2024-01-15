import logging
import math
import numpy as np
from torch.autograd import Variable
from fedscale.cloud.execution.torch_client import TorchClient


class FedAvgTrainer(TorchClient):
    def __init__(self, conf):

        super(FedAvgTrainer, self).__init__(conf)
        self.global_model = None


    def train(self, client_data, model, conf):
        """
        Perform a training task.
        :param client_data: client training dataset
        :param model: the framework-specific model
        :param conf: job config
        :return: training results
        """
        total_data_num = conf.total_data
        # logging.info(total_data_num)

        cur_data_num = len(client_data) * client_data.batch_size
        client_id = conf.client_id
        logging.info(f"Start to train (CLIENT: {client_id}) ...")

        model = model.to(device=self.device)
        model.train()

        trained_unique_samples = min(
            len(client_data.dataset), conf.local_steps * conf.batch_size)
        # logging.info(hasattr(conf, 'epochs'))
        epochs = conf.epochs
        optimizer = self.get_optimizer(model, conf)
        criterion = self.get_criterion(conf)
        error_type = None
        self.global_model = None
        # NOTE: If one may hope to run fixed number of epochs, instead of iterations,
        # use `while self.completed_steps < conf.local_steps * len(client_data)` instead
        while self.completed_steps < epochs:

            self.train_step(client_data, conf, model, optimizer, criterion)


        state_dicts = model.state_dict()
        model_param = {p: state_dicts[p].data.cpu().numpy()
                       for p in state_dicts}

        logging.info(f"{cur_data_num}, {total_data_num}, {cur_data_num / total_data_num}")
        model_param = [cur_data_num / total_data_num * x.astype(np.float64) for x in model_param.values()]

        results = {'client_id': client_id, 'moving_loss': self.epoch_train_loss,
                   'trained_size': self.completed_steps * conf.batch_size,
                   'success': self.completed_steps == conf.local_steps}


        logging.info(f"Training of (CLIENT: {client_id}) completes, {results}")


        results['utility'] = math.sqrt(
            self.loss_squared) * float(trained_unique_samples)
        results['update_weight'] = model_param
        results['wall_duration'] = 0

        return results
