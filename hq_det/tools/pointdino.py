"""PointDINO configuration and compatibility entry points for HQTrainer."""

import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys


def _default_config():
    return str(
        Path(__file__).resolve().parents[1] / 'models' / 'pointdino' /
        'configs' / 'pointdino_r50_shanghaitech_stage4_12e.py')


def _parse_args(argv=None, *, evaluation=False):
    from mmengine.config import DictAction

    description = 'Evaluate PointDINO' if evaluation else 'Train PointDINO'
    parser = argparse.ArgumentParser(description=description)
    if evaluation:
        parser.add_argument('config', help='PointDINO config file')
        parser.add_argument('checkpoint', help='Checkpoint to evaluate')
    else:
        parser.add_argument(
            'config', nargs='?', default=_default_config(),
            help='PointDINO config file (default: full-map Stage 4)')
        initialization = parser.add_mutually_exclusive_group()
        initialization.add_argument(
            '--load-from', metavar='CHECKPOINT',
            help='Initialize model weights, without resuming optimizer state')
        initialization.add_argument(
            '--resume', nargs='?', const='auto', metavar='CHECKPOINT',
            help='Resume a checkpoint, or the latest checkpoint in work-dir')
        parser.add_argument(
            '--amp', action='store_true', help='Enable automatic mixed precision')
        parser.add_argument(
            '--auto-scale-lr', action='store_true',
            help='Scale learning rate to the actual total batch size')

    parser.add_argument('--work-dir', help='Directory for logs and checkpoints')
    parser.add_argument(
        '--data-root', help='Point dataset root; updates all data loaders')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='Config overrides as key=value, including nested keys and lists')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch'], default='none',
        help='HQ distributed launcher; use torchrun for multiple GPUs')
    parser.add_argument(
        '--local-rank', '--local_rank', type=int, default=0,
        help='Local process rank supplied by distributed launchers')
    return parser.parse_args(argv)


def _prepare_config(args, *, evaluation=False):
    from mmengine.config import Config
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    cfg.launcher = args.launcher
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif not cfg.get('work_dir'):
        cfg.work_dir = str(Path('work_dirs') / Path(args.config).stem)

    if args.data_root:
        cfg.data_root = args.data_root
        for name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
            if cfg.get(name) is not None:
                cfg[name].dataset.data_root = args.data_root

    # The shared HQ runtime uses fork, which is unavailable on Windows.
    if sys.platform == 'win32':
        cfg.setdefault('env_cfg', {}).setdefault('mp_cfg', {})[
            'mp_start_method'] = 'spawn'

    if evaluation:
        cfg.load_from = args.checkpoint
        cfg.resume = False
    else:
        if args.load_from:
            cfg.load_from = args.load_from
            cfg.resume = False
        if args.resume:
            cfg.resume = True
            cfg.load_from = None if args.resume == 'auto' else args.resume
        if args.amp:
            wrapper_type = cfg.optim_wrapper.get('type', 'OptimWrapper')
            if wrapper_type not in ('OptimWrapper', 'AmpOptimWrapper'):
                raise ValueError('--amp requires OptimWrapper or AmpOptimWrapper')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')
        if args.auto_scale_lr:
            if 'base_batch_size' not in cfg.get('auto_scale_lr', {}):
                raise ValueError('--auto-scale-lr requires auto_scale_lr.base_batch_size')
            cfg.auto_scale_lr.enable = True

    return cfg


def train_main(argv=None):
    """Run the detector's loss path, optimizer, scheduler and validation loop."""
    args = _parse_args(argv)
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    cfg = _prepare_config(args)
    from .train_pointdino import PointDINOTrainer

    trainer = PointDINOTrainer.from_config(cfg)
    trainer.run()
    return trainer


def evaluate_main(argv=None):
    """Evaluate point predictions using the configured PointDINOMetric."""
    args = _parse_args(argv, evaluation=True)
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    cfg = _prepare_config(args, evaluation=True)
    from .train_pointdino import PointDINOTrainer

    return PointDINOTrainer.from_config(cfg).test()


def _dataset_config(cfg, name, data_path, ann_file, image_dir, class_names,
                    image_size, batch_size, num_workers, category_ids=None):
    """Configure a split without assuming a ShanghaiTech directory layout."""
    annotation = Path(ann_file)
    if not annotation.is_absolute():
        annotation = Path(data_path) / annotation
    if not annotation.is_file():
        raise FileNotFoundError(f'PointDINO annotation does not exist: {annotation}')
    from hq_det.pointdino_data import read_point_annotations

    content = read_point_annotations(annotation, class_names=class_names,
                                     category_ids=category_ids)
    metadata = content['metainfo']
    dataset = cfg[name].dataset
    dataset.update(data_root=str(data_path), ann_file=str(ann_file),
                   data_prefix=dict(img_path=str(image_dir)),
                   metainfo=metadata)
    cfg[name].update(batch_size=batch_size, num_workers=num_workers,
                     persistent_workers=num_workers > 0)
    # Native pixels by default. Explicit resizing must transform training GT,
    # while evaluation GT stays in original coordinates for rescaled outputs.
    pipeline = [transform for transform in dataset.pipeline
                if transform.type != 'PointDINOResize']
    if image_size is not None:
        annotation_index = next(
            index for index, transform in enumerate(pipeline)
            if transform.type == 'PointDINOLoadAnnotations')
        insertion_index = (annotation_index + 1 if name == 'train_dataloader'
                           else annotation_index)
        pipeline.insert(insertion_index, dict(
            type='PointDINOResize', scale=image_size, keep_ratio=False))
    dataset.pipeline = pipeline
    return metadata


def build_run_config(
        data_path, output_path, *, load_checkpoint=None, config_path=None,
        train_ann_file=None, val_ann_file=None,
        test_ann_file=None, train_image_dir=None,
        val_image_dir=None, test_image_dir=None, class_names=None,
        image_size=None, batch_size=2, eval_batch_size=1,
        num_data_workers=2, num_epoches=12, lr0=1e-4,
        lr_backbone_mult=0.1, gradient_update_interval=1, lr_milestones=None,
        resume=False, score_threshold=0.5, distance_thresholds=(5., 10.),
        cfg_options=None, evaluation=False):
    """Build the full source-style configuration for a custom point dataset.

    ``image_size=None`` preserves original image dimensions. An explicit
    (width, height) pair or integer enables resizing (a square for an integer).
    COCO category IDs are sorted and mapped to contiguous training labels;
    legacy JSON labels follow ``metainfo.classes``. Full-model
    checkpoints must have the same class count and architecture.
    ``cfg_options`` is applied last for advanced MMEngine overrides.
    """
    from mmengine import Config
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(str(config_path or _default_config()))
    # Default to the same Roboflow/COCO directory layout as other HQ models.
    # Existing source point datasets remain usable without rewriting JSON.
    legacy_layout = (Path(data_path) / 'train_point.json').is_file() or (
        (Path(data_path) / 'test_point.json').is_file()
        and not (Path(data_path) / 'valid' / '_annotations.coco.json').is_file())
    train_ann_file = train_ann_file or ('train_point.json' if legacy_layout else 'train/_annotations.coco.json')
    val_ann_file = val_ann_file or ('test_point.json' if legacy_layout else 'valid/_annotations.coco.json')
    if train_image_dir is None:
        train_image_dir = 'train_data/images' if legacy_layout else 'train'
    if val_image_dir is None:
        val_image_dir = 'test_data/images' if legacy_layout else 'valid'
    if image_size is not None:
        image_size = ((image_size, image_size) if isinstance(image_size, int)
                      else tuple(image_size))
        if len(image_size) != 2 or any(size <= 0 for size in image_size):
            raise ValueError('image_size must be None or a positive (width, height) pair.')
    if min(batch_size, eval_batch_size, num_epoches, gradient_update_interval) < 1:
        raise ValueError('Batch sizes, epochs and gradient interval must be positive.')
    if num_data_workers < 0:
        raise ValueError('num_data_workers must be nonnegative.')
    if not distance_thresholds or any(x <= 0 or int(x) != x for x in distance_thresholds):
        raise ValueError('distance_thresholds must contain positive integer pixels.')
    # Config inheritance can alias val/test dictionaries; keep splits separate.
    for name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        cfg[name] = deepcopy(cfg[name])
    cfg.val_evaluator = deepcopy(cfg.val_evaluator)
    cfg.test_evaluator = deepcopy(cfg.test_evaluator)
    if evaluation:
        metadata = _dataset_config(
            cfg, 'test_dataloader', data_path, test_ann_file or val_ann_file,
            val_image_dir if test_image_dir is None else test_image_dir,
            class_names, image_size, eval_batch_size, num_data_workers)
    else:
        metadata = _dataset_config(
            cfg, 'train_dataloader', data_path, train_ann_file, train_image_dir,
            class_names, image_size, batch_size, num_data_workers)
        _dataset_config(cfg, 'val_dataloader', data_path, val_ann_file,
                        val_image_dir, metadata['classes'], image_size, eval_batch_size,
                        num_data_workers, category_ids=metadata['category_ids'])
        cfg.test_dataloader = deepcopy(cfg.val_dataloader)
        if test_ann_file is not None:
            _dataset_config(
                cfg, 'test_dataloader', data_path, test_ann_file,
                val_image_dir if test_image_dir is None else test_image_dir,
                metadata['classes'], image_size, eval_batch_size, num_data_workers,
                category_ids=metadata['category_ids'])
    cfg.model.bbox_head.num_classes = len(metadata['classes'])
    cfg.data_root = str(data_path)
    cfg.work_dir = str(output_path)
    cfg.launcher = 'pytorch' if int(os.environ.get('WORLD_SIZE', '1')) > 1 else 'none'
    if sys.platform == 'win32':
        cfg.env_cfg.mp_cfg.mp_start_method = 'spawn'
    if load_checkpoint:
        if not Path(load_checkpoint).is_file():
            raise FileNotFoundError(f'PointDINO checkpoint does not exist: {load_checkpoint}')
        cfg.load_from = str(load_checkpoint)
        # Runner initializes before loading; the full checkpoint already has
        # the backbone, so do not download a redundant torchvision checkpoint.
        cfg.model.backbone.init_cfg = None
    else:
        cfg.load_from = None
    if evaluation and not load_checkpoint:
        raise ValueError('Testing requires load_checkpoint.')
    cfg.resume = bool(resume) if not evaluation else False
    if cfg.resume:
        cfg.model.backbone.init_cfg = None
    cfg.max_epochs = num_epoches
    cfg.train_cfg.max_epochs = num_epoches
    milestones = (list(lr_milestones) if lr_milestones is not None
                  else [num_epoches - 1] if num_epoches > 1 else [])
    cfg.param_scheduler = [dict(type='MultiStepLR', begin=0, end=num_epoches,
                               by_epoch=True, milestones=milestones, gamma=0.1)]
    cfg.optim_wrapper.optimizer.lr = lr0
    cfg.optim_wrapper.paramwise_cfg.custom_keys.backbone.lr_mult = lr_backbone_mult
    cfg.optim_wrapper.accumulative_counts = gradient_update_interval
    for name in ('val_evaluator', 'test_evaluator'):
        cfg[name].score_threshold = score_threshold
        cfg[name].distance_thresholds = list(distance_thresholds)
    best_threshold = 10 if 10.0 in distance_thresholds else int(distance_thresholds[0])
    cfg.default_hooks.checkpoint.save_best = f'point/f1@{best_threshold}px'
    if cfg_options:
        cfg.merge_from_dict(cfg_options)
    return cfg


def run(data_path, output_path='output/pointdino', *, devices=None, **kwargs):
    """Train from explicit parameters, like HQ's existing train_dino.run."""
    from .train_pointdino import run as train

    return train(data_path, output_path, devices=devices, **kwargs)


def test(data_path, load_checkpoint, output_path='output/pointdino_test', *,
         devices=None, **kwargs):
    """Evaluate a checkpoint against an independently configured test split."""
    from .train_pointdino import test as evaluate

    return evaluate(data_path, load_checkpoint, output_path, devices=devices, **kwargs)
