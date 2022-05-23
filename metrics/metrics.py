from abc import abstractmethod

import torch
from torch import nn

from utils import check_zero_divide, sum_except_dim


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


class BalancedAccuracy(_ByClassMetric):

    def get_batch_metric(self, predictions, targets):
        correct = sum_except_dim(predictions * targets, dim=1)
        total = sum_except_dim(targets, dim=1)
        self._corrects += correct.type(self._corrects.dtype)
        self._totals += total

        return check_zero_divide(correct, total)


class MeanIoU(_ByClassMetric):

    def get_batch_metric(self, predictions, targets):
        intersect = sum_except_dim(torch.mul(predictions, targets), dim=1)
        union = sum_except_dim(torch.maximum(predictions, targets), dim=1)

        self._corrects += intersect.type(self._corrects.dtype)
        self._totals += union.type(self._totals.dtype)

        return intersect / union
