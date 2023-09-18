from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import flwr as fl




# pylint: disable=unsubscriptable-object,bad-option-value,R1725