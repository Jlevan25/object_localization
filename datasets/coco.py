import json
import os
import pickle
from typing import Tuple, Any, List

import numpy as np
from PIL.Image import Image
from torchvision.datasets import CocoDetection
from metrics import ImageMetric


class CocoLocalizationDataset(CocoDetection):

    def __init__(self, root: str, annFile: str, mapping: str = None, image_metrics: List[ImageMetric] = None,
                 transform=None, target_transform=None, transforms=None):
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.overfit = False
        self._batch_size = None
        self.mapping = None
        self.image_metrics = image_metrics

        if mapping is not None:
            self.mapping = self._get_mapping(mapping)
            self.class_names = {v['mapped_id']: v['name'] for v in self.mapping.values()}
        else:
            self.class_names = {k: v['name'] for k, v in self.coco.cats.items()}

        if image_metrics is not None:
            for metric in image_metrics:
                metric.set_class_names(self.class_names)

    @staticmethod
    def _get_mapping(mapping_path):
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
            f.close()
        return {int(k): v for k, v in mapping.items()}

    def set_overfit_mode(self, batch_size: int):
        self.overfit = True
        self._batch_size = batch_size

    def unset_overfit_mode(self):
        self.overfit = False

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.image_metrics is not None:

            for metric in self.image_metrics:
                metric.append_image(image, target)

        target = target[0]['category_id']

        if self.mapping is not None:
            target = self.mapping[target]['mapped_id']

        if self.transforms is not None:
            image, mapped_target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids) if not self.overfit else self._batch_size
