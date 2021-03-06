from abc import abstractmethod, ABC

import torch

from utils import check_zero_divide


class Metric:

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
        self.class_names = None

    def set_class_names(self, class_names:dict):
        self.class_names = class_names

    def append_image(self, image, target):
        self.batch_images.append(image)
        self.batch_targets.append(target)

    def _clear_batch(self):
        self.batch_images = []
        self.batch_targets = []
