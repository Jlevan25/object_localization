import os.path

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import models
from torch import optim, nn
from torchvision import transforms, datasets

from datasets import CocoLocalizationDataset
from executors.trainer import Trainer
from transforms import UnNormalize, GetFromAnns
from configs import Config
from metrics import MeanIoU, BalancedAccuracy
from torch.utils.tensorboard import SummaryWriter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(ROOT, 'datasets', 'COCO_SOLO')

cfg = Config(ROOT_DIR=ROOT, DATASET_DIR=DATASET_ROOT,
             dataset_name='COCO_SOLO', out_features=90,
             model_name='Resnet50', device='cpu',
             batch_size=128, lr=0.01, overfit=False,
             debug=True, show_each=100,
             seed=None)

train_key, valid_key, test_key = 'train', 'val', 'test'

if cfg.seed is not None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

resnet_preprocess_block = [transforms.RandomResizedCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],  # resnet imagnet
                                                std=[0.229, 0.224, 0.225])]

train_preprocess = transforms.Compose(resnet_preprocess_block)
eval_preprocess = transforms.Compose(resnet_preprocess_block)

# target_preprocess = transforms.Compose([GetFromAnns(category=True)])
# target_preprocess = transforms.Compose([])
target_preprocess = None

transform = {train_key: train_preprocess, valid_key: eval_preprocess, test_key: eval_preprocess}
target_transform = {train_key: target_preprocess, valid_key: target_preprocess, test_key: target_preprocess}

datasets_dict = {train_key: CocoLocalizationDataset(root=os.path.join(cfg.DATASET_DIR, train_key),
                                                    annFile=os.path.join(cfg.DATASET_DIR, 'annotations',
                                                                         f'instances_{train_key}.json'),
                                                    transform=transform[train_key],
                                                    target_transform=target_transform[train_key]),

                 valid_key: CocoLocalizationDataset(root=os.path.join(cfg.DATASET_DIR, valid_key),
                                                    annFile=os.path.join(cfg.DATASET_DIR, 'annotations',
                                                                         f'instances_{valid_key}.json'),
                                                    transform=transform[valid_key],
                                                    target_transform=target_transform[valid_key])}

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

metrics = [BalancedAccuracy(cfg.out_features)]
model = models.resnet50(pretrained=True,
                        num_classes=cfg.out_features).to(cfg.device)

optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
scheduler = ReduceLROnPlateau(optimizer)
criterion = nn.CrossEntropyLoss()

writer = SummaryWriter(log_dir=cfg.LOG_PATH)

trainer = Trainer(datasets_dict=datasets_dict, dataloaders_dict=dataloaders_dict,
                  model=model, optimizer=optimizer, scheduler=scheduler,
                  criterion=criterion, writer=writer, cfg=cfg, metrics=metrics)

# trainer.load_model(trainer_cfg.LOAD_PATH)
epoch = 5
for epoch in range(epoch):
    # trainer.fit(epoch)
    # trainer.save_model(epoch)

    trainer.writer.add_scalar(f'scheduler lr', trainer.optimizer.param_groups[0]['lr'], epoch)
    trainer.validation(epoch)
