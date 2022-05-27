import os
import sys

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from executors import EpochManager, PhaseKeysDict
from configs import Config, Resnet50Config, BottleneckBlockConfig
from models import Resnet
from metrics import BalancedAccuracy
from datasets import CustomImageFolder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(ROOT, 'datasets')

cfg = Config(ROOT_DIR=ROOT, DATASET_DIR=DATASET_ROOT,
             dataset_name='AlmostCifar', out_features=12,
             model_name='Resnet50', device='cuda',
             batch_size=128, lr=0.01, overfit=False,
             debug=True, show_each=100,
             seed=None)

model_cfg = Resnet50Config(in_channels=3, out_features=cfg.out_features)

train_key, valid_key, test_key = 'train', 'valid', 'test'

if cfg.seed is not None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

jitter_param = (0.6, 1.4)
norm = [transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.405],
                             std=[0.229, 0.224, 0.225])]

transforms = {train_key: transforms.Compose([transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ColorJitter(brightness=jitter_param,
                                                                    saturation=jitter_param,
                                                                    hue=(-.25, .25)),
                                             *norm]),

              valid_key: transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             *norm])}

datasets_dict = {train_key: CustomImageFolder(root=os.path.join(DATASET_ROOT, train_key),
                                              transform=transforms[train_key]),
                 valid_key: CustomImageFolder(root=os.path.join(DATASET_ROOT, valid_key),
                                              transform=transforms[valid_key])}

if cfg.overfit:
    shuffle = False
    for dataset in datasets_dict.values():
        dataset.set_overfit_mode(cfg.batch_size)
else:
    shuffle = True

dataloaders_dict = {train_key: DataLoader(datasets_dict[train_key],
                                          batch_size=cfg.batch_size, shuffle=shuffle),
                    valid_key: DataLoader(datasets_dict[valid_key],
                                          batch_size=cfg.batch_size)}

model = Resnet(model_cfg).to(cfg.device)

# weight decay
wd_params, no_wd_params = model.split_params4weight_decay()
params = [dict(params=wd_params, weight_decay=1e-4),
          dict(params=no_wd_params)]

optimizer = optim.Adam(params, lr=cfg.lr)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer)
writer = SummaryWriter(log_dir=cfg.LOG_PATH)
metrics = [BalancedAccuracy(model_cfg.out_features)]

epoch_manager = EpochManager(model=model,
                             criterion=criterion,
                             optimizer=optimizer,
                             writer=writer,
                             dataloaders=dataloaders_dict,
                             cfg=cfg,
                             scheduler=scheduler,
                             metrics=metrics,
                             phase_keys=PhaseKeysDict(train_key, valid_key, test_key))

epochs = 20

for epoch in range(epochs):
    epoch_manager.train(epoch)
    epoch_manager.save_model(epoch)

    for i, param_group in enumerate(epoch_manager.optimizer.param_groups):
        epoch_manager.writer.add_scalar(f'scheduler lr/param_group{i}',
                                        param_group['lr'], epoch)
    epoch_manager.validation(epoch)
