from abc import abstractmethod, ABC

import torch
from torch import nn

from utils import check_zero_divide


class Metric(nn.Module):

    def forward(self, epoch: bool = False, *args, **kwargs):
        return self.get_epoch_metric() if epoch else self.get_batch_metric(*args, **kwargs)

    @abstractmethod
    def get_epoch_metric(self):
        raise NotImplementedError()

    @abstractmethod
    def get_batch_metric(self, predictions, targets):
        raise NotImplementedError()


class _ByClassMetric(Metric, ABC):
    def __init__(self, classes: int):
        super().__init__()
        self.classes = classes
        self._corrects = torch.tensor([0. for _ in range(self.classes)])
        self._totals = torch.tensor([0. for _ in range(self.classes)])

    def get_epoch_metric(self):
        mean = check_zero_divide(self._corrects, self._totals)
        self._corrects *= 0
        self._totals *= 0
        return mean


class ImageMetric(Metric, ABC):
    def __init__(self):
        super().__init__()
        self.batch_images = []
        self.batch_targets = []

    def append_image(self, image, target):
        self.batch_images.append(image)
        self.batch_targets.append(target)

    def _clear_batch(self):
        self.batch_images = []
        self.batch_targets = []
