import os

import json
import numpy as np
from PIL import Image

from configs import Config
from datasets import CocoLocalizationDataset


def filter(dataset, stage):
    indexes = [i for i, target in enumerate([dataset._load_target(cur_id) for cur_id in dataset.ids]) if target]
    ids = np.array(dataset.ids)[indexes]
    annotations = []
    images = []
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(ROOT, 'COCO', stage)

    if not os.path.exists(path):
        os.makedirs(path)

    for idishnik in ids.tolist():
        image = dataset.coco.loadImgs(idishnik)[0]["file_name"]
        target = dataset.coco.loadAnns(dataset.coco.getAnnIds(idishnik))

        result = Image.open(os.path.join(ROOT, 'datasets', 'COCO_SOLO', stage, image))
        result.save(os.path.join(path, image))
        images.append(dataset.coco.loadImgs(idishnik)[0])
        annotations.extend(target)

    dataset.coco.dataset['images'] = images
    dataset.coco.dataset['annotations'] = annotations

    path = os.path.join(ROOT, 'COCO', 'annotations')

    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(path, f'instances_{stage}.json')

    print('create.json')
    with open(path, 'w') as f:
        json.dump(dataset.coco.__dict__['dataset'], f)


if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(ROOT_DIR, 'datasets', 'COCO_SOLO')

    cfg = Config(ROOT_DIR=ROOT_DIR, DATASET_DIR=DATASET_DIR,
                 dataset_name='COCO_SOLO', out_features=80,
                 model_name='Resnet50', device='cpu',
                 batch_size=128, lr=0.01, overfit=False,
                 debug=True, show_each=100,
                 seed=None)

    train_key, valid_key, test_key = 'train', 'val', 'test'

    datasets_dict = {train_key: CocoLocalizationDataset(root=os.path.join(cfg.DATASET_DIR, train_key),
                                                        annFile=os.path.join(cfg.DATASET_DIR, 'annotations',
                                                                             f'instances_{train_key}.json')),

                     valid_key: CocoLocalizationDataset(root=os.path.join(cfg.DATASET_DIR, valid_key),
                                                        annFile=os.path.join(cfg.DATASET_DIR, 'annotations',
                                                                             f'instances_{valid_key}.json'))}
    filter(datasets_dict[train_key], train_key)
    filter(datasets_dict[valid_key], valid_key)
