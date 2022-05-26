import torch

from metrics.metrics import _ByClassMetric

from utils import check_zero_divide, sum_except_dim

from torch.nn import functional as F
from utils import one_hot_argmax


class BalancedAccuracy(_ByClassMetric):

    def get_batch_metric(self, predictions, targets):
        predictions = F.one_hot(predictions, num_classes=self.classes).transpose(1, -1).squeeze(-1)
        targets = F.one_hot(targets, num_classes=self.classes).transpose(1, -1).squeeze(-1)

        correct = sum_except_dim(predictions * targets, dim=1)
        total = sum_except_dim(targets, dim=1)
        self._corrects += correct.type(self._corrects.dtype)
        self._totals += total.type(self._totals.dtype)

        return check_zero_divide(correct, total)


class MeanIoU(_ByClassMetric):

    def get_batch_metric(self, predictions, targets):
        predictions = F.one_hot(predictions, num_classes=self.classes).transpose(1, -1).squeeze(-1)
        targets = F.one_hot(targets, num_classes=self.classes).transpose(1, -1).squeeze(-1)

        intersect = sum_except_dim(torch.mul(predictions, targets), dim=1)
        union = sum_except_dim(torch.maximum(predictions, targets), dim=1)

        self._corrects += intersect.type(self._corrects.dtype)
        self._totals += union.type(self._totals.dtype)

        return intersect / union
