from hq_det.models.rfdetr.detr import RFDETRBase
from hq_det.models.rfdetr.config import RFDETRBaseConfig


model = RFDETRBase(
    pretrain_weights="/root/autodl-tmp/model/rfdetr/rf-detr-base.pth"
)

model.train(
    epochs=10,
    batch_size=4,
    lr=1e-4,
    lr_encoder=1.5e-4,
    weight_decay=1e-4,
    grad_accum_steps=4,
    num_workers=4,
    resolution=1024,
    dataset_dir="/root/autodl-tmp/chengdu_v2_dataset",
)