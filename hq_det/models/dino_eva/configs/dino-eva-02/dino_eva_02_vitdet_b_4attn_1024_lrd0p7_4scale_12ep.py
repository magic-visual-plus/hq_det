from functools import partial
from detrex.config import get_config
from detrex.modeling.backbone.eva import get_vit_lr_decay_rate

from ..common.coco_loader_lsj_1024 import dataloader
from ..models.dino_eva_02 import model

# modify model config
model.backbone.net.img_size = 1024 
model.backbone.square_pad = 1024  
model.backbone.net.patch_size = 16  
model.backbone.net.window_size = 16 
model.backbone.net.embed_dim = 768
model.backbone.net.depth = 12
model.backbone.net.num_heads = 12
model.backbone.net.mlp_ratio = 4*2/3
model.backbone.net.use_act_checkpoint = False
model.backbone.net.drop_path_rate = 0.1

# 2, 5, 8, 11 for global attention
model.backbone.net.window_block_indexes = [0, 1, 3, 4, 6, 7, 9, 10]




