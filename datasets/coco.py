from typing import Tuple, Any

from torchvision.datasets import CocoDetection


class CocoLocalizationDataset(CocoDetection):

    def __init__(self, root: str, annFile: str, transform=None, target_transform=None, transforms=None):
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.overfit = True
        self._batch_size = None

    def set_overfit_mode(self, batch_size: int):
        self.overfit = True
        self._batch_size = batch_size

    def unset_overfit_mode(self):
        self.overfit = False

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)[0]['category_id']

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids) if not self.overfit else self._batch_size
