from abc import abstractmethod

import torch
from torch import nn
from utils import check_zero_divide


class _ByClassMetric(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.classes = classes
        self._corrects = torch.tensor([0 for _ in range(self.classes)])
        self._totals = torch.tensor([0 for _ in range(self.classes)])

    def forward(self, epoch: bool = False, *args, **kwargs):
        return self.get_epoch_metric() if epoch else self.get_batch_metric(*args, **kwargs)

    def get_epoch_metric(self):
        mean = check_zero_divide(self._corrects, self._totals)
        self._corrects *= 0
        self._totals *= 0
        return mean

    @abstractmethod
    def get_batch_metric(self, predictions, targets):
        raise NotImplementedError()
