import os
import torch
import os.path as osp
from torch.utils import data
from torchnet.meter import ConfusionMeter, AverageValueMeter

m = ConfusionMeter(3)
p = torch.tensor([[0.3, 0.7, 0.1], [0.7, 0.2, 0.9], [0.9, 0.2, 1]])
t = torch.tensor([0, 1, 1])
m.add(p, t)

n = m.value()
pass