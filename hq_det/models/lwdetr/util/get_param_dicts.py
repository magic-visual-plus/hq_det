# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Functions to get params dict"""
import torch.nn as nn

from hq_det.models.lwdetr.models.backbone import Joiner


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
    print("name: {}, lr_decay: {}".format(name, lr_decay_rate ** (num_layers + 1 - layer_id)))
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_vit_weight_decay_rate(name, weight_decay_rate=1.0):
    if ('gamma' in name) or ('pos_embed' in name) or ('rel_pos' in name) or ('bias' in name) or ('norm' in name):
        weight_decay_rate = 0.
    print("name: {}, weight_decay rate: {}".format(name, weight_decay_rate))
    return weight_decay_rate


def get_param_dict(args, model_without_ddp: nn.Module):
    """
    获取模型参数的优化器配置字典
    
    Args:
        args: 训练参数配置
        model_without_ddp: 不带分布式数据并行的模型
    
    Returns:
        final_param_dicts: 包含所有参数组的优化器配置列表
    """
    # 确保backbone是Joiner类型
    assert isinstance(model_without_ddp.backbone, Joiner)
    
    # 获取backbone部分
    backbone = model_without_ddp.backbone[0]
    # 获取backbone的参数和学习率配置对
    backbone_named_param_lr_pairs = backbone.get_named_param_lr_pairs(args, prefix="backbone.0")
    # 提取backbone的参数配置字典列表
    backbone_param_lr_pairs = [param_dict for _, param_dict in backbone_named_param_lr_pairs.items()]

    # 获取decoder部分的参数
    decoder_key = 'transformer.decoder'
    decoder_params = [
        p
        for n, p in model_without_ddp.named_parameters() if decoder_key in n and p.requires_grad
    ]

    # 为decoder参数配置学习率(使用衰减系数)
    decoder_param_lr_pairs = [
        {"params": param, "lr": args.lr * args.lr_component_decay} 
        for param in decoder_params
    ]

    # 获取其他参数(既不在backbone也不在decoder中的参数)
    other_params = [
        p
        for n, p in model_without_ddp.named_parameters() if (
            n not in backbone_named_param_lr_pairs and decoder_key not in n and p.requires_grad)
    ]
    # 为其他参数配置基础学习率
    other_param_dicts = [
        {"params": param, "lr": args.lr} 
        for param in other_params
    ]
    
    # 合并所有参数配置
    final_param_dicts = (
        other_param_dicts + backbone_param_lr_pairs + decoder_param_lr_pairs
    )

    return final_param_dicts
