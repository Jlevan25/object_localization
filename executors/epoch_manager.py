import os
from typing import Union

import torch
from torch.nn import functional as F
from utils import one_hot_argmax


class PhaseKeysDict(dict):
    def __init__(self, train: str = 'train', validation: str = 'valid', test: str = 'test'):
        super().__init__(train=train, valid=validation, test=test)


class EpochManager(object):
    def __init__(self, model, criterion, optimizer, writer, dataloaders: dict, cfg, scheduler,
                 metrics: list = None, phase_keys: Union[PhaseKeysDict, list] = None):
        self.model = model
        self.model.to(cfg.device)

        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.scheduler = scheduler

        self.writer = writer
        self.metrics = metrics

        self.cfg = cfg
        self.device = self.cfg.device

        if phase_keys is not None:
            self._phase_keys = PhaseKeysDict(*phase_keys) if isinstance(phase_keys, list) else phase_keys
        else:
            self._phase_keys = PhaseKeysDict()

        self._global_step = {v: 0 for v in phase_keys.values()}

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
        checkpoint = torch.load(path)
        self._global_step = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print('model loaded')

    def train(self, i_epoch):
        self.model.train()
        self._step(self._phase_keys['train'])

    @torch.no_grad()
    def validation(self, i_epoch):
        self.model.eval()
        self._step(self._phase_keys['valid'], i_epoch)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self._step(self._phase_keys['test'])

    def _step(self, phase, epoch=None):

        calc_metrics = self.metrics is not None and self.metrics
        print('\n_______', phase, f'epoch{epoch}' if epoch is not None else '',
              'len:', len(self.dataloaders[phase]), '_______')

        for i, (images, targets) in enumerate(self.dataloaders[phase]):

            debug = self.cfg.debug and i % self.cfg.show_each == 0

            if debug:
                print('\n___', f'Iteration {i}', '___')

            predictions = self.model(images.to(self.device))

            one_hots = F.one_hot(targets, num_classes=self.cfg.out_features)

            if calc_metrics:
                self._calc_batch_metrics(predictions, one_hots, phase, debug)

            if phase == 'train':
                loss = self.criterion(predictions, one_hots.float().to(self.device))
                self.writer.add_scalar(f'{phase}/loss', loss, self._global_step[phase])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss.detach())

                if debug:
                    print(f'Train Loss: {loss.item()}')

            self._global_step[phase] += 1

        if calc_metrics and epoch is not None:
            print('\n___', f'Epoch Summary', '___')
            self._calc_epoch_metrics(phase)

    def _calc_epoch_metrics(self, phase):
        self._calc_metrics(phase, self.cfg.debug, is_epoch=True)

    def _calc_batch_metrics(self, predictions, targets, phase, debug):
        self._calc_metrics(phase, debug, one_hot_argmax(predictions), targets)

    @torch.no_grad()
    def _calc_metrics(self, phase, debug, *batch, is_epoch: bool = False):
        for metric in self.metrics:
            values = metric(is_epoch, *batch).tolist()
            metric_name = type(metric).__name__

            if len(values) > 1:
                for cls, scalar in (zip(self.classes, values) if hasattr(self, 'classes') else enumerate(values)):
                    self.writer.add_scalar(f'{phase}/{metric_name}/{cls}', scalar, self._global_step[phase])

            self.writer.add_scalar(f'{phase}/{metric_name}/average',
                                   sum(values) / len(values), self._global_step[phase])

            if debug:
                print("{}: {}".format(metric_name, values))

    def __repr__(self) -> str:
        return '{}(\n\tmodel={},\n\tcriterion={},\n\toptimizer={})'.format(
            type(self).__name__, str(self.model).replace('\n\t', ''), self.criterion, self.optimizer
        )
