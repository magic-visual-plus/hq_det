from mmdet.configs.dino import dino_4scale_r50_8xb2_12e_coco as dino_config
from mmengine import MODELS

if __name__ == '__main__':

    model_config = dino_config.model
    # create model
    model = MODELS.build(model_config)

    print(type(model))
    print(model.loss)
    pass