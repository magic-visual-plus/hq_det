_base_ = ['/root/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py']

pretrained = '/root/autodl-tmp/model/codetr/swin_large_patch4_window12_384_22k.pth'  # noqa
load_from = '/root/autodl-tmp/model/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(
            checkpoint=pretrained
        )
    )
)


