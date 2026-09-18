"""PointDINO's model interface for the shared HQ training lifecycle."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import warnings

import numpy as np
import torch
from mmengine.registry import DefaultScope

from mmdet.registry import MODELS
from mmdet.utils import register_all_modules

from ..base import HQModel
from .pointdino_transforms import PointDINOPackDetInputs, PointDINOResize


class PointDINOModel(HQModel):
    """Expose the complete point detector through the existing HQModel API.

    Training executes the detector's loss method, including the optional FIDT
    branch. Evaluation returns DetDataSample objects with point predictions.
    Checkpoints retain the detector's original parameter names.
    """

    def __init__(self,
                 cfg,
                 class_id2names=None,
                 image_size=None,
                 checkpoint=None):
        super().__init__(class_id2names=class_id2names)
        self.cfg = deepcopy(cfg)
        self.image_size = image_size
        self.loaded_checkpoint = None
        self.missing_checkpoint_keys = []
        checkpoint_data = (self._read_checkpoint(checkpoint)
                           if checkpoint is not None else None)
        checkpoint_classes = self._checkpoint_classes(checkpoint_data)
        if class_id2names is None:
            classes = checkpoint_classes
            if classes is None:
                dataset = self.cfg.get('train_dataloader',
                                       {}).get('dataset', {})
                classes = dataset.get('metainfo', {}).get('classes')
            if classes is None:
                classes = [
                    str(index)
                    for index in range(self.cfg.model.bbox_head.num_classes)
                ]
            class_id2names = dict(enumerate(classes))
        self.id2names = dict(class_id2names)
        if sorted(self.id2names) != list(range(len(self.id2names))):
            raise ValueError(
                'PointDINO class IDs must be contiguous from zero.')
        if not self.id2names:
            raise ValueError('PointDINO requires at least one class.')
        self.num_classes = len(self.id2names)
        if (checkpoint_classes is not None
                and len(checkpoint_classes) != self.num_classes):
            raise ValueError(
                'Checkpoint class count mismatch: '
                f'checkpoint={len(checkpoint_classes)}, '
                f'dataset={self.num_classes}.')
        self.cfg.model.bbox_head.num_classes = self.num_classes
        model_cfg = deepcopy(self.cfg.model)
        if checkpoint_data is not None:
            self._disable_pretrained(model_cfg)
        register_all_modules(init_default_scope=False)
        # BaseModel constructs its preprocessor through MMEngine's parent
        # registry. Limit the MMDetection scope to this construction only.
        with DefaultScope.overwrite_default_scope('mmdet'):
            self.model = MODELS.build(model_cfg)
        self.model.init_weights()
        self.device = next(self.model.parameters()).device
        if checkpoint_data is not None:
            self.load_model(checkpoint_data)

    @staticmethod
    def _read_checkpoint(checkpoint):
        if isinstance(checkpoint, Mapping):
            return checkpoint
        data = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if not isinstance(data, Mapping):
            raise ValueError('PointDINO checkpoint must contain a state dict.')
        return data

    @staticmethod
    def _checkpoint_classes(checkpoint):
        if checkpoint is None:
            return None
        metadata = checkpoint.get('meta', {})
        dataset_meta = metadata.get('dataset_meta', {})
        classes = dataset_meta.get('classes', dataset_meta.get('CLASSES'))
        if classes is None:
            classes = metadata.get('CLASSES')
        return list(classes) if classes is not None else None

    @staticmethod
    def _disable_pretrained(config):
        """Avoid downloading initialization weights before a complete restore."""
        if isinstance(config, Mapping):
            if 'pretrained' in config:
                config['pretrained'] = None
            initializers = config.get('init_cfg')
            if isinstance(initializers, Mapping):
                if initializers.get('type') == 'Pretrained':
                    config['init_cfg'] = None
            elif isinstance(initializers, (list, tuple)):
                config['init_cfg'] = [
                    item for item in initializers
                    if not (isinstance(item, Mapping)
                            and item.get('type') == 'Pretrained')
                ] or None
            for value in config.values():
                PointDINOModel._disable_pretrained(value)
        elif isinstance(config, (list, tuple)):
            for value in config:
                PointDINOModel._disable_pretrained(value)

    def get_class_names(self):
        return [self.id2names[index] for index in range(self.num_classes)]

    def load_model(self, checkpoint):
        """Restore compatible weights and return the full resume payload.

        A Stage 3 checkpoint may omit the entire optional FIDT branch. All
        other missing, unexpected or incompatible parameters are rejected
        before changing model weights.
        """
        data = self._read_checkpoint(checkpoint)
        state = data.get('state_dict', data.get('model'))
        if state is None and all(torch.is_tensor(v) for v in data.values()):
            state = data
        if not isinstance(state, Mapping) or not state:
            raise ValueError(
                'Checkpoint does not contain a nonempty state_dict.')
        state = dict(state)
        for prefix in ('module.', 'model.'):
            if all(name.startswith(prefix) for name in state):
                state = {
                    name[len(prefix):]: value
                    for name, value in state.items()
                }
        expected = self.model.state_dict()
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        mismatched = [
            name for name in set(expected) & set(state)
            if (not torch.is_tensor(state[name])
                or state[name].shape != expected[name].shape)
        ]
        disallowed_missing = [
            name for name in missing if not name.startswith('point_fidt_head.')
        ]
        fidt_keys = {
            name
            for name in expected if name.startswith('point_fidt_head.')
        }
        if set(missing) & fidt_keys and fidt_keys & set(state):
            raise ValueError(
                'Checkpoint contains only part of the FIDT auxiliary branch.')
        if disallowed_missing or unexpected or mismatched:
            raise ValueError(
                'Incompatible PointDINO checkpoint: '
                f'missing={disallowed_missing}, unexpected={unexpected}, '
                f'shape_mismatch={sorted(mismatched)}')
        self.model.load_state_dict(state, strict=False)
        self.missing_checkpoint_keys = missing
        if missing:
            warnings.warn(
                'Checkpoint has no weights for these FIDT auxiliary '
                f'parameters; their initialization is retained: {missing}',
                UserWarning,
                stacklevel=2)
        self.loaded_checkpoint = data
        return data

    def forward(self, batch_data):
        processed = self.model.data_preprocessor(batch_data,
                                                 training=self.training)
        return self.model(**processed,
                          mode='loss' if self.training else 'predict')

    def compute_loss(self, batch_data, forward_result):
        if not isinstance(forward_result, Mapping):
            return torch.zeros((), device=self.device), {'loss': 0.0}
        # MMEngine sums only keys containing 'loss'. Detached FIDT diagnostics
        # remain available for logging without entering the objective twice.
        loss, log_values = self.model.parse_losses(forward_result)
        info = {
            name: float(value.detach().cpu())
            for name, value in log_values.items()
        }
        return loss, info

    def postprocess(self, forward_result, batch_data=None, confidence=0.0):
        if confidence <= 0:
            return forward_result
        predictions = []
        for sample in forward_result:
            sample = sample.clone()
            sample.pred_instances = sample.pred_instances[
                sample.pred_instances.scores >= confidence]
            predictions.append(sample)
        return predictions

    def checkpoint_dict(self):
        return {
            'state_dict': self.model.state_dict(),
            'meta': {
                'dataset_meta': {
                    'classes': self.get_class_names(),
                    'CLASSES': self.get_class_names(),
                }
            },
            'image_size': self.image_size,
            'model_config': deepcopy(self.cfg.model),
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_dict(), path)

    def to(self, *args, **kwargs):
        torch.nn.Module.to(self, *args, **kwargs)
        # Moving a parent nn.Module does not call its children's to overrides.
        # MMEngine's BaseModel.to also updates the preprocessor's cast device.
        self.model.to(*args, **kwargs)
        self.device = next(self.model.parameters()).device
        return self

    @torch.no_grad()
    def predict(self, imgs, bgr=False, confidence=0.0, image_size=None):
        """Predict points from RGB arrays, or BGR arrays with ``bgr=True``.

        Images retain their dimensions unless image_size is configured. The
        returned points always use original-image pixel coordinates.
        """
        if not imgs:
            return []
        size = self.image_size if image_size is None else image_size
        if size is not None:
            size = (size, size) if isinstance(size, int) else tuple(size)
            if len(size) != 2 or min(size) <= 0:
                raise ValueError(
                    'image_size must be a positive width/height pair.')
        pack = PointDINOPackDetInputs()
        resize = PointDINOResize(scale=size,
                                 keep_ratio=False) if size else None
        packed = []
        for image_id, img in enumerate(imgs):
            image = np.asarray(img)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError('Images must have shape [height, width, 3].')
            image = image.copy() if bgr else image[..., ::-1].copy()
            results = dict(img=image,
                           img_id=image_id,
                           img_shape=image.shape[:2],
                           ori_shape=image.shape[:2],
                           scale_factor=(1.0, 1.0),
                           gt_points=np.empty((0, 2), dtype=np.float32),
                           gt_points_labels=np.empty((0, ), dtype=np.int64))
            if resize is not None:
                results = resize(results)
            packed.append(pack(results))
        batch = dict(inputs=[item['inputs'] for item in packed],
                     data_samples=[item['data_samples'] for item in packed])
        was_training = self.training
        self.eval()
        try:
            return self.postprocess(self(batch), confidence=confidence)
        finally:
            self.train(was_training)
