import json
import os.path
import pickle

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import models
from torch import optim, nn
from torchvision import transforms, datasets

from datasets import CocoLocalizationDataset
from executors.trainer import Trainer
from transforms import UnNormalize, GetFromAnns, ClassMapping
from configs import Config
from metrics import MeanIoU, BalancedAccuracy
from torch.utils.tensorboard import SummaryWriter
from metrics import ClassAP
from utils import split_params4weight_decay

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(ROOT, 'datasets', 'COCO_SOLO')

cfg = Config(ROOT_DIR=ROOT, DATASET_DIR=DATASET_ROOT,
             dataset_name='COCO_SOLO', out_features=80,
             model_name='Resnet50', device='cuda',
             batch_size=64, lr=5e-4, overfit=False,
             debug=True, show_each=100,
             seed=None)

train_key, valid_key, test_key = 'train', 'val', 'test'

if cfg.seed is not None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

model = models.resnet50(pretrained=True,
                        num_classes=cfg.out_features).to(cfg.device)

image_metrics = [ClassAP(num_classes=cfg.out_features, iou_threshold=0.5, confidence_threshold=0.5,
                         with_cam=True, cam_layer=model.layer4[-1], fc_layer=model.fc)]
# save_path=os.path.join(ROOT, 'pred_bboxes', 'my'))]

train_metrics = [BalancedAccuracy(cfg.out_features)]
eval_metrics = [*train_metrics, *image_metrics]

metrics = {train_key: train_metrics, valid_key: eval_metrics, test_key: eval_metrics}

resnet_preprocess_block = [transforms.RandomResizedCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],  # resnet imagnet
                                                std=[0.229, 0.224, 0.225])]

train_preprocess = transforms.Compose(resnet_preprocess_block)
eval_preprocess = transforms.Compose(resnet_preprocess_block)

target_preprocess = None

transform = {train_key: train_preprocess, valid_key: eval_preprocess, test_key: eval_preprocess}
target_transform = {train_key: target_preprocess, valid_key: target_preprocess, test_key: target_preprocess}

mapping_path = os.path.join(DATASET_ROOT, 'annotations', 'mapping.json')
datasets_dict = {train_key: CocoLocalizationDataset(root=os.path.join(cfg.DATASET_DIR, train_key),
                                                    annFile=os.path.join(cfg.DATASET_DIR, 'annotations',
                                                                         f'instances_{train_key}.json'),
                                                    mapping=mapping_path,
                                                    image_metrics=image_metrics,
                                                    transform=transform[train_key],
                                                    target_transform=target_transform[train_key]),

                 valid_key: CocoLocalizationDataset(root=os.path.join(cfg.DATASET_DIR, valid_key),
                                                    annFile=os.path.join(cfg.DATASET_DIR, 'annotations',
                                                                         f'instances_{valid_key}.json'),
                                                    mapping=mapping_path,
                                                    image_metrics=image_metrics,
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

wd_params, no_wd_params = split_params4weight_decay(model)
params = [dict(params=wd_params, weight_decay=5e-4),
          dict(params=no_wd_params)]

optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer)
criterion = nn.CrossEntropyLoss()

writer = SummaryWriter(log_dir=cfg.LOG_PATH)

trainer = Trainer(datasets_dict=datasets_dict, dataloaders_dict=dataloaders_dict,
                  model=model, optimizer=optimizer, scheduler=scheduler,
                  criterion=criterion, writer=writer, cfg=cfg, metrics=metrics)

trainer.load_model(os.path.join(ROOT, 'checkpoints', '8.pth'))
epoch = 1
for epoch in range(epoch):
    trainer.fit(epoch)
    trainer.save_model(epoch)

    trainer.writer.add_scalar(f'scheduler lr', trainer.optimizer.param_groups[0]['lr'], epoch)
    trainer.validation(epoch)
