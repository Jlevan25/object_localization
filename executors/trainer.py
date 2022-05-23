import os.path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import models
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from datasets import CocoLocalizationDataset
from utils import one_hot_argmax


class Trainer:
    def __init__(self,
                 model, optimizer,
                 scheduler, criterion,
                 writer, cfg,
                 datasets_dict=None,
                 dataloader_dict=None,
                 transform: dict = None,
                 target_transform: dict = None,
                 metrics: list = None):
        self.cfg = cfg
        self.model = models.resnet50(pretrained=True,
                                     num_classes=self.cfg.model_params.out_features).to(self.cfg.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, **self.cfg.scheduler_kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = metrics
        self.transform = transform
        self.target_transform = target_transform
        self.device = self.cfg.device
        self.writer = writer

        self.datasets = datasets_dict if datasets_dict is not None else dict()
        self.dataloaders = dataloader_dict if dataloader_dict is not None else dict()

        self._global_step = dict()

    def _get_data(self, data_type):
        if data_type not in self._global_step:
            self._global_step[data_type] = -1
        transform = self.transform[data_type] if self.transform is not None else None
        target_transform = self.target_transform[data_type] if self.target_transform is not None else None

        self.datasets[data_type] = CocoLocalizationDataset(root=os.path.join(self.cfg.DATASET_PATH, data_type),
                                                           annFile=os.path.join(self.cfg.DATASET_PATH, 'annotations',
                                                                                f'instances_{data_type}.json'),
                                                           transform=transform,
                                                           target_transform=target_transform)

        def collate_target_dict(batch):
            img_list, target_list = batch
            target = target_list[0]
            for t in target_list[1:]:
                for k, v in t.items():
                    target[k] = target[k]
            return torch.stack(img_list, 0), target

        self.dataloaders[data_type] = DataLoader(self.datasets[data_type], batch_size=self.cfg.batch_size,
                                                 collate_fn=collate_target_dict, shuffle=self.cfg.shuffle)

    @torch.no_grad()
    def _calc_epoch_metrics(self, stage):
        self._calc_metrics(stage, self.cfg.debug, is_epoch=True)

    @torch.no_grad()
    def _calc_batch_metrics(self, masks, targets, stage, debug):
        self._calc_metrics(stage, debug, one_hot_argmax(masks), targets)

    def _calc_metrics(self, stage, debug, *batch, is_epoch: bool = False):
        for metric in self.metrics:
            values = metric(is_epoch, *batch).tolist()
            metric_name = type(metric).__name__

            for cls, scalar in (zip(self.classes, values) if hasattr(self, 'classes') else enumerate(values)):
                self.writer.add_scalar(f'{stage}/{metric_name}/{cls}', scalar, self._global_step[stage])

            self.writer.add_scalar(f'{stage}/{metric_name}/overall',
                                   sum(values) / len(values), self._global_step[stage])

            if debug:
                print("{}: {}".format(metric_name, values))

    def _epoch_step(self, stage='test', epoch=None):

        if stage not in self.dataloaders:
            self._get_data(stage)

        calc_metrics = self.metrics is not None and self.metrics
        print('\n_______', stage, f'epoch{epoch}' if epoch is not None else '',
              'len:', len(self.dataloaders[stage]), '_______')

        for i, (images, targets) in enumerate(self.dataloaders[stage]):

            self._global_step[stage] += 1
            debug = self.cfg.debug and i % self.cfg.show_each == 0

            if debug:
                print('\n___', f'Iteration {i}', '___')

            one_hots = F.one_hot(targets, num_classes=self.cfg.model_params.out_features).transpose(1, -1).squeeze(-1)
            predictions = self.model(images.to(self.device))

            if calc_metrics:
                self._calc_batch_metrics(predictions, one_hots, stage, debug)

            if stage == 'train':
                loss = self.criterion(predictions, one_hots.float().to(self.device))
                self.writer.add_scalar(f'{stage}/loss', loss, self._global_step[stage])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss.detach())

                if debug:
                    print(f'Train Loss: {loss.item()}')

        if calc_metrics and epoch is not None:
            print('\n___', f'Epoch Summary', '___')
            self._calc_epoch_metrics(stage)

    def fit(self, i_epoch):
        self._epoch_step(stage='train', epoch=i_epoch)

    @torch.no_grad()
    def validation(self, i_epoch):
        self._epoch_step(stage='val', epoch=i_epoch)

    @torch.no_grad()
    def test(self):
        self._epoch_step(stage='test')

    def save_model(self, epoch):
        path = os.path.join(self.cfg.SAVE_PATH, f'{epoch}.pth')

        if not os.path.exists(self.cfg.SAVE_PATH):
            os.makedirs(self.cfg.SAVE_PATH)

        torch.save(self.model.state_dict(), path)
        print('model saved')

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print('model loaded')
