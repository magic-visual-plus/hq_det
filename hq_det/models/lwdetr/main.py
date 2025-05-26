# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
cleaned main file
"""
import argparse
import datetime
import json
import random
import time
import ast
import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from util.drop_scheduler import drop_scheduler
from util.get_param_dicts import get_param_dict
import util.misc as utils
from util.utils import ModelEma, BestMetricHolder, clean_state_dict
from util.benchmark import benchmark


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_encoder', default=1.5e-4, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=12, type=int)
    parser.add_argument('--lr_drop', default=11, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--lr_vit_layer_decay', default=0.8, type=float)
    parser.add_argument('--lr_component_decay', default=1.0, type=float)

    # drop args 
    # dropout and stochastic depth drop rate; set at most one to non-zero
    parser.add_argument('--dropout', type=float, default=0,
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--drop_path', type=float, default=0,
                        help='Drop path rate (default: 0.0)')

    # early / late dropout and stochastic depth settings
    parser.add_argument('--drop_mode', type=str, default='standard',
                        choices=['standard', 'early', 'late'], help='drop mode')
    parser.add_argument('--drop_schedule', type=str, default='constant',
                        choices=['constant', 'linear'],
                        help='drop schedule for early dropout / s.d. only')
    parser.add_argument('--cutoff_epoch', type=int, default=0,
                        help='if drop_mode is early / late, this is the epoch where dropout ends / starts')

    # Model parameters
    parser.add_argument('--pretrained_encoder', type=str, default=None, 
                        help="Path to the pretrained encoder.")
    parser.add_argument('--pretrain_weights', type=str, default=None, 
                        help="Path to the pretrained model.")
    parser.add_argument('--pretrain_exclude_keys', type=str, default=None, nargs='+', 
                        help="Keys you do not want to load.")
    parser.add_argument('--pretrain_keys_modify_to_load', type=str, default=None, nargs='+',
                        help="Keys you want to modify to load. Only used when loading objects365 pre-trained weights.")

    # * Backbone
    parser.add_argument('--encoder', default='vit_tiny', type=str,
                        help="Name of the transformer or convolutional encoder to use")
    parser.add_argument('--vit_encoder_num_layers', default=12, type=int,
                        help="Number of layers used in ViT encoder")
    parser.add_argument('--window_block_indexes', default=None, type=int, nargs='+')
    parser.add_argument('--position_embedding', default='sine', type=str, 
                        choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--out_feature_indexes', default=[-1], type=int, nargs='+', help='only for vit now')

    # * Transformer
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--sa_nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's self-attentions")
    parser.add_argument('--ca_nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's cross-attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--group_detr', default=13, type=int,
                        help="Number of groups to speed up detr training")
    parser.add_argument('--two_stage', action='store_true')
    parser.add_argument('--projector_scale', default='P4', type=str, nargs='+', choices=('P3', 'P4', 'P5', 'P6'))
    parser.add_argument('--lite_refpoint_refine', action='store_true', help='lite refpoint refine mode for speed-up')
    parser.add_argument('--num_select', default=100, type=int,
                        help='the number of predictions selected for evaluation')
    parser.add_argument('--dec_n_points', default=4, type=int,
                        help='the number of sampling points')
    parser.add_argument('--decoder_norm', default='LN', type=str)
    parser.add_argument('--bbox_reparam', action='store_true')

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--sum_group_losses', action='store_true',
                        help="To sum losses across groups or mean losses.")
    parser.add_argument('--use_varifocal_loss', action='store_true')
    parser.add_argument('--use_position_supervised_loss', action='store_true')
    parser.add_argument('--ia_bce_loss', action='store_true')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--square_resize_div_64', action='store_true')

    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoint_interval', default=10, type=int,
                        help='epoch interval to save checkpoint')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--ema_decay', default=0.9997, type=float)
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--sync_bn', default=True, type=bool,
                        help='setup synchronized BatchNorm for distributed training')
    
    # fp16
    parser.add_argument('--fp16_eval', default=False, action='store_true',
                        help='evaluate in fp16 precision.')

    # subparsers
    subparsers = parser.add_subparsers(title='sub-commands', dest='subcommand',
        description='valid subcommands', help='additional help')

    # subparser for export model
    parser_export = subparsers.add_parser('export_model', help='LWDETR model export')
    parser_export.add_argument('--shape', type=int, nargs=2, default=(640, 640), help="input shape (width, height)")
    parser_export.add_argument('--infer_dir', type=str, default=None)
    parser_export.add_argument('--verbose', type=ast.literal_eval, default=False, nargs="?", const=True)
    parser_export.add_argument('--opset_version', type=int, default=17)
    parser_export.add_argument('--simplify', action='store_true', help="Simplify onnx model")
    parser_export.add_argument('--tensorrt', '--trtexec', '--trt', action='store_true',
                               help="build tensorrt engine")
    parser_export.add_argument('--dry-run', '--test', '-t', action='store_true', help="just print command")
    return parser


def main(args):
    # 初始化分布式训练模式
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    # 设置设备
    device = torch.device(args.device)

    # 设置随机种子以确保可重复性
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 构建模型、损失函数和后处理器
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    
    # 如果使用EMA(指数移动平均),则初始化EMA模型
    if args.use_ema:
        ema_m = ModelEma(model, decay=args.ema_decay)
    else:
        ema_m = None
        
    # 处理分布式训练
    model_without_ddp = model
    if args.distributed:
        if args.sync_bn:
            # 转换为同步BatchNorm
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # 使用DistributedDataParallel进行分布式训练
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        
    # 计算模型参数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dicts = get_param_dict(args, model_without_ddp)

    # 初始化优化器和学习率调度器
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, 
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # 构建训练和验证数据集
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    # 设置数据采样器
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # 创建批次采样器
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    # 创建数据加载器
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, 
                                 num_workers=args.num_workers)

    # 获取COCO API用于评估
    base_ds = get_coco_api_from_dataset(dataset_val)

    # 加载预训练权重
    if args.pretrain_weights is not None:
        checkpoint = torch.load(args.pretrain_weights, map_location='cpu')
        # 支持排除某些键
        # 例如,加载object365预训练时,不加载`class_embed.[weight, bias]`
        if args.pretrain_exclude_keys is not None:
            assert isinstance(args.pretrain_exclude_keys, list)
            for exclude_key in args.pretrain_exclude_keys:
                checkpoint['model'].pop(exclude_key)
        if args.pretrain_keys_modify_to_load is not None:
            from util.obj365_to_coco_model import get_coco_pretrain_from_obj365
            assert isinstance(args.pretrain_keys_modify_to_load, list)
            for modify_key_to_load in args.pretrain_keys_modify_to_load:
                checkpoint['model'][modify_key_to_load] = get_coco_pretrain_from_obj365(
                    model_without_ddp.state_dict()[modify_key_to_load],
                    checkpoint['model'][modify_key_to_load]
                )
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if args.use_ema:
            del ema_m
            ema_m = ModelEma(model_without_ddp)

    # 设置输出目录
    output_dir = Path(args.output_dir)
    
    # 在主进程上进行基准测试
    if utils.is_main_process():
        print("Get benchmark")
        benchmark_model = copy.deepcopy(model_without_ddp)
        bm = benchmark(benchmark_model.float(), dataset_val, output_dir)
        print(json.dumps(bm, indent=2))
        del benchmark_model
    
    # 恢复训练
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model) 
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            checkpoint['optimizer']["param_groups"] = optimizer.state_dict()["param_groups"]
            checkpoint['lr_scheduler'].pop("step_size")
            checkpoint['lr_scheduler'].pop("_last_lr")
            
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    # 如果只进行评估
    if args.eval:
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return
    
    # 设置dropout和drop path的调度器
    total_batch_size = args.batch_size * utils.get_world_size()
    num_training_steps_per_epoch = (len(dataset_train) + total_batch_size - 1) // total_batch_size
    schedules = {}
    if args.dropout > 0:
        schedules['do'] = drop_scheduler(
            args.dropout, args.epochs, num_training_steps_per_epoch,
            args.cutoff_epoch, args.drop_mode, args.drop_schedule)
        print("Min DO = %.7f, Max DO = %.7f" % (min(schedules['do']), max(schedules['do'])))

    if args.drop_path > 0:
        schedules['dp'] = drop_scheduler(
            args.drop_path, args.epochs, num_training_steps_per_epoch,
            args.cutoff_epoch, args.drop_mode, args.drop_schedule)
        print("Min DP = %.7f, Max DP = %.7f" % (min(schedules['dp']), max(schedules['dp'])))

    # 开始训练
    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    
    # 训练循环
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:    # 分布式训练
            sampler_train.set_epoch(epoch)
            
        # 训练一个epoch
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, ema_m=ema_m, schedules=schedules, 
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            vit_encoder_num_layers=args.vit_encoder_num_layers, args=args)
            
        # 计算训练时间
        train_epoch_time = time.time() - epoch_start_time
        train_epoch_time_str = str(datetime.timedelta(seconds=int(train_epoch_time)))
        
        # 更新学习率
        lr_scheduler.step()
        
        # 保存检查点
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # 在LR下降前和每`checkpoint_interval`个epoch保存额外检查点
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if args.use_ema:
                    weights.update({
                        'ema_model': ema_m.module.state_dict(),
                    })
                utils.save_on_master(weights, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args=args
        )
        
        map_regular = test_stats['coco_eval_bbox'][0]
        _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
        if _isbest:
            checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if args.use_ema:
            ema_test_stats, _ = evaluate(
                ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args=args
            )
            log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats.items()})
            map_ema = ema_test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
            if _isbest:
                checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                utils.save_on_master({
                    'model': ema_m.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        log_stats.update(best_map_holder.summary())
        
        # epoch parameters
        ep_paras = {
                'epoch': epoch,
                'n_parameters': n_parameters
            }
        log_stats.update(ep_paras)
        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass
        log_stats['train_epoch_time'] = train_epoch_time_str
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LWDETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.subcommand is None:
        main(args)
    elif args.subcommand == 'export_model':
        from deploy.export import main
        if args.batch_size != 1:
            args.batch_size = 1
            print(f"Only batch_size 1 is supported for onnx export, \
                 but got batchsize = {args.batch_size}. batch_size is forcibly set to 1.")
        main(args)
