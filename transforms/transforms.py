import torch
from torch import Tensor, nn


class UnNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """

        mean = torch.tensor(self.mean).reshape((1, -1, 1, 1))
        std = torch.tensor(self.std).reshape((1, -1, 1, 1))
        return tensor * std + mean


class GetFromAnns(nn.Module):
    def __init__(self, category: bool = True, bounding_box: bool = False, mask: bool = False):
        super().__init__()
        self.category = category
        self.bb = bounding_box
        self.mask = mask
        self._out_dict = dict()

    def _get_category(self, anns):
        self._out_dict['category'] = anns

    def _get_mask(self, anns):
        self._out_dict['mask'] = anns

    def _get_bb(self, anns):
        self._out_dict['bb'] = anns

    def forward(self, anns):
        if self.category:
            self._get_category(anns)

        if self.mask:
            self._get_mask(anns)

        if self.bb:
            self._get_bb(anns)

        return self._out_dict


class ToDict(nn.Module):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys
        self.dict = dict()

    def forward(self, target: list):
        for i, t in enumerate(target):
            self.dict[self.keys[i]] = t
