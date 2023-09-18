from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from typing import Dict, List, Optional, Tuple
import numpy as np
import flwr as fl
import util
class FlowerClient(fl.client.Client):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        print(f"[client {self.cid}] get parameters")

        ndarrays:List[np.ndarray]  = util.get_parameters(self.net)

        parameters = ndarrays_to_parameters(ndarrays)
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters
        )
    
    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.cid}] fit, config : {ins.config}")
        # Deserialize parameters to NumPy ndarrays's
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)
        # Upload local model, train, get updated parameters
        util.set_parameters(self.net, ndarrays_original)
        util.train(self.net, self.trainloader, epochs=1)
        ndarrays_updated = util.get_parameters(self.net)
        parameters_updated = parameters_to_ndarrays(ndarrays_updated)
        status = Status(code=Code.Ok, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(self.trainloader),
            metrics={},
        )
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"[Client {self.cid}] evaluate, config: {ins.config}")

        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)

        util.set_parameters(self.net, ndarrays_original)
        loss, accuracy = util.test(self.net, self.valloader)
        # return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.valloader),
            metrics={"accuracy": float(accuracy)},
        )


def client_fn(cid) -> FlowerClient:
    model = util.load_model()
    trainset, testset = util.load_data()
    return FlowerClient(cid, model, trainset, testset)
        