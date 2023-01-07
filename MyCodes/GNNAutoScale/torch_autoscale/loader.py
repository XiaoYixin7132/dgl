from typing import NamedTuple, List, Tuple

import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from dgl import DGLGraph
from dgl.heterograph import DGLBlock
