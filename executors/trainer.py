import os.path
from typing import Iterator

import torch
from torch import tensor
from torch.utils.data import DataLoader
from datasets import CocoLocalizationDataset


class Trainer:
    def __init__(self,
                 model, optimizer,
                 scheduler, criterion,
                 writer, cfg,
                 class_names=None,
                 dataloaders_dict=None,
                 transform: dict = None,
                 target_transform: dict = None,
                 metrics: dict = None):
        self.cfg = cfg
        self.model = model

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.metrics = metrics
        self.transform = transform
        self.target_transform = target_transform
        self.device = self.cfg.device
        self.writer = writer

        #self.datasets = datasets_dict if datasets_dict is not None else dict()
        self.class_names = class_names if class_names is not None else list(range(cfg.out_features))
        self.dataloaders = dataloaders_dict if dataloaders_dict is not None else dict()

        self._global_step = dict()

    def _get_global_step(self, data_type):
        self._global_step[data_type] = -1

    def _get_data(self, data_type):
        transform = self.transform[data_type] if self.transform is not None else None
        target_transform = self.target_transform[data_type] if self.target_transform is not None else None

        self.datasets[data_type] = CocoLocalizationDataset(root=os.path.join(self.cfg.DATASET_PATH, data_type),
                                                           annFile=os.path.join(self.cfg.DATASET_PATH, 'annotations',
                                                                                f'instances_{data_type}.json'),
                                                           transform=transform,
                                                           target_transform=target_transform)

        self.dataloaders[data_type] = DataLoader(self.datasets[data_type], batch_size=self.cfg.batch_size,
                                                 shuffle=self.cfg.shuffle)

    @torch.no_grad()
    def _calc_epoch_metrics(self, stage):
        for metric in self.metrics[stage]:
            self._write_metrics(metric_values=metric.get_epoch_metric(),
                                metric_name=type(metric).__name__,
                                stage=stage, debug=self.cfg.debug)

    @torch.no_grad()
    def _calc_batch_metrics(self, predictions, targets, stage, debug):
        for metric in self.metrics[stage]:
            self._write_metrics(metric_values=metric.get_batch_metric(predictions, targets),
                                metric_name=type(metric).__name__,
                                stage=stage, debug=debug)

    def _write_metrics(self, metric_values, metric_name, stage, debug):

        if len(metric_values) > 1:
            if self.cfg.write_by_class_metrics:
                for cls, scalar in zip(self.class_names, metric_values):
                    if scalar >= 0:
                        self.writer.add_scalar(f'{stage}/{metric_name}/{cls}', scalar.item(), self._global_step[stage])

            mean_value = metric_values[metric_values >= 0]
            mean_value = mean_value.mean().item() if len(mean_value) > 0 else 0.
            self.writer.add_scalar(f'{stage}/{metric_name}/mean', mean_value, self._global_step[stage])

            if debug:
                metric_values[metric_values < 0] = 0.
                if self.cfg.write_by_class_metrics:
                    print("{}: {}".format(metric_name, list(metric_values)))

                print("Mean {}: {}".format(metric_name, mean_value))

        else:
            metric_values = metric_values.item()
            self.writer.add_scalar(f'{stage}/{metric_name}', metric_values, self._global_step[stage])

            if debug:
                print("{}: {}".format(metric_name, metric_values))


    def _epoch_generator(self, stage, epoch=None) -> Iterator[tensor]:

        if stage not in self.dataloaders:
            self._get_data(stage)

        if stage not in self._global_step:
            self._get_global_step(stage)

        calc_metrics = self.metrics[stage] is not None and len(self.metrics[stage]) > 0
        print('\n_______', stage, f'epoch{epoch}' if epoch is not None else '',
              'len:', len(self.dataloaders[stage]), '_______')

        for i, (images, targets) in enumerate(self.dataloaders[stage]):

            self._global_step[stage] += 1
            debug = self.cfg.debug and i % self.cfg.show_each == 0

            predictions = self.model(images.to(self.device))

            loss = self.criterion(predictions, targets.to(self.device))
            self.writer.add_scalar(f'{stage}/loss', loss, self._global_step[stage])

            if debug:
                print('\n___', f'Iteration {i}', '___')
                print(f'Loss: {loss.item()}')

            if calc_metrics:
                self._calc_batch_metrics(predictions.argmax(1), targets.cpu(), stage, debug)

            yield loss

        if calc_metrics:
            print('\n___', f'Epoch Summary', '___')
            self._calc_epoch_metrics(stage)

    def fit(self, stage_key, i_epoch):
        for batch_loss in self._epoch_generator(stage=stage_key, epoch=i_epoch):
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def validation(self, stage_key, i_epoch):
        for batch_loss in self._epoch_generator(stage=stage_key, epoch=i_epoch):
            self.scheduler.step(batch_loss.detach())

    @torch.no_grad()
    def test(self, stage_key):
        self._epoch_generator(stage=stage_key)

    def save_model(self, epoch, path=None):
        path = self.cfg.SAVE_PATH if path is None else path

        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, f'{epoch}.pth')

        checkpoint = dict(epoch=self._global_step,
                          model=self.model.state_dict(),
                          optimizer=self.optimizer.state_dict())

        torch.save(checkpoint, path)
        print('model saved, epoch:', epoch)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=torch.device(self.device))
        self._global_step = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print('model loaded')
