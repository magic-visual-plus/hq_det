"""Point datasets plugged into HQ's existing dataset and provider interfaces."""

from copy import deepcopy
import os

import numpy as np
from mmcv.transforms import LoadImageFromFile
from mmengine.dataset import Compose
from pycocotools.coco import COCO
from torchvision.datasets import VisionDataset

from .dataset import CocoDetection
from .dataset_provider import HQDatasetProvider
from .pointdino_data import read_point_annotations


class PointDINODataset(CocoDetection):
    """HQ dataset accepting COCO-point and legacy PointDINO annotations.

    The inherited image loader, dataset length and class mapping are reused.
    COCO's in-memory image index supplies filenames to that image loader;
    point annotations remain in ``data_list`` and are never converted to boxes.

    Each item contains ``inputs`` (a CHW tensor), ``data_samples`` (a
    DetDataSample) and ``image_id``. Collate these as lists so the model's
    data preprocessor can pad images of different native sizes.
    """

    def __init__(self, img_folder, ann_file, transforms=None, metainfo=None,
                 test_mode=False, max_refetch=1000):
        # CocoDetection.__init__ expects a COCO annotation file. Initialize its
        # common torchvision base, then supply the index its image loader uses.
        VisionDataset.__init__(self, root=os.fspath(img_folder))
        self.ann_file = os.fspath(ann_file)
        self.test_mode = bool(test_mode)
        self.max_refetch = int(max_refetch)
        if self.max_refetch < 0:
            raise ValueError('max_refetch must be nonnegative.')
        content = read_point_annotations(
            self.ann_file, class_names=(metainfo or {}).get('classes'),
            category_ids=(metainfo or {}).get('category_ids'))
        self.annotation_format = content['format']
        self.metainfo = deepcopy(content['metainfo'])
        self.metainfo.update(deepcopy(metainfo or {}))
        classes = self.metainfo['classes']
        self.metainfo['classes'] = tuple(classes)
        self.metainfo['category_ids'] = tuple(self.metainfo['category_ids'])
        self.id2names = dict(enumerate(classes))
        self.labels = []

        self.data_list = deepcopy(content['data_list'])
        self.ids = []
        images = []
        for record in self.data_list:
            image_id = record['img_id']
            self.ids.append(image_id)
            images.append(dict(id=image_id, file_name=record['img_path']))
        self.coco = COCO()
        self.coco.dataset = dict(
            images=images, annotations=[],
            categories=[dict(id=i, name=name) for i, name in self.id2names.items()])
        self.coco.createIndex()
        self._transforms = self._compose_pipeline(transforms)
        self.transforms = self._transforms

    @staticmethod
    def _compose_pipeline(transforms):
        from mmdet.registry import TRANSFORMS
        # Register the point-specific transforms without changing native ones.
        from .models import pointdino  # noqa: F401

        if transforms is None:
            transforms = [dict(type='PointDINOLoadAnnotations'),
                          dict(type='PointDINOPackDetInputs')]
        elif hasattr(transforms, 'transforms'):
            transforms = transforms.transforms
        elif callable(transforms):
            transforms = [transforms]
        pipeline = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = deepcopy(transform)
                kind = transform.get('type')
                if (kind is LoadImageFromFile
                        or isinstance(kind, str)
                        and kind.rsplit('.', 1)[-1] == 'LoadImageFromFile'):
                    continue
                transform = TRANSFORMS.build(transform)
            if isinstance(transform, LoadImageFromFile):
                continue
            pipeline.append(transform)
        return Compose(pipeline)

    @classmethod
    def from_config(cls, dataset_cfg, transforms=None):
        """Adapt a BaseDetDataset config to HQ's PointDINO dataset."""
        config = deepcopy(dict(dataset_cfg))
        config.pop('type', None)
        data_root = os.fspath(config.pop('data_root', ''))
        ann_file = os.fspath(config.pop('ann_file'))
        if not os.path.isabs(ann_file):
            ann_file = os.path.join(data_root, ann_file)
        data_prefix = config.pop('data_prefix', dict(img_path=''))
        if set(data_prefix) != {'img_path'}:
            raise ValueError('PointDINO data_prefix must contain only img_path.')
        img_folder = os.fspath(data_prefix['img_path'])
        if not os.path.isabs(img_folder):
            img_folder = os.path.join(data_root, img_folder)
        pipeline = config.pop('pipeline', None)
        if transforms is None:
            transforms = pipeline
        # These MMEngine storage options do not alter sample contents.
        config.pop('serialize_data', None)
        config.pop('lazy_init', None)
        if config.pop('backend_args', None) is not None:
            raise ValueError('HQ PointDINO image loading supports local files.')
        if config.pop('filter_cfg', None):
            raise ValueError('PointDINO does not use bounding-box dataset filters.')
        return cls(img_folder, ann_file, transforms=transforms, **config)

    def __getitem__(self, index):
        for _ in range(self.max_refetch + 1):
            image_id = self.ids[index]
            image = self._load_image(image_id)
            if image is None:
                path = os.path.join(self.root, self.data_list[index]['img_path'])
                raise ValueError(f'Cannot decode PointDINO image: {path}')
            result = deepcopy(self.data_list[index])
            height, width = image.shape[:2]
            result.update(
                img=image, img_path=os.path.join(self.root, result['img_path']),
                img_shape=(height, width), ori_shape=(height, width),
                height=height, width=width, sample_idx=index)
            result = self._transforms(result)
            if result is not None:
                result['image_id'] = image_id
                return result
            if self.test_mode:
                raise RuntimeError('PointDINO test pipeline discarded a sample.')
            index = np.random.randint(len(self))
        raise RuntimeError('PointDINO pipeline exceeded max_refetch attempts.')


class PointDINODatasetProvider(HQDatasetProvider):
    """Build point datasets through the same provider hooks as HQ detectors."""

    def __init__(self, train_config=None, val_config=None, test_config=None):
        super().__init__()
        self.train_config = deepcopy(train_config)
        self.val_config = deepcopy(val_config)
        self.test_config = deepcopy(test_config)
        self._class_metainfo = None

    def _build(self, config, transforms, split):
        if config is None:
            raise ValueError(f'No PointDINO {split} dataset configuration supplied.')
        dataset = PointDINODataset.from_config(config, transforms=transforms)
        metadata = (dataset.metainfo['category_ids'], dataset.metainfo['classes'])
        if self._class_metainfo is None:
            self._class_metainfo = metadata
        elif self._class_metainfo != metadata:
            raise ValueError(f'PointDINO {split} category ID-to-name mapping '
                             'differs from the other dataset splits.')
        return dataset

    def build_train_dataset(self, transforms=None):
        return self._build(self.train_config, transforms, 'training')

    def build_valid_dataset(self, transforms=None):
        return self._build(self.val_config, transforms, 'validation')

    def build_test_dataset(self, transforms=None):
        return self._build(self.test_config, transforms, 'test')
