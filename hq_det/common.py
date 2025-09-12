import pydantic
import numpy as np
from typing import List
from . import box_utils


class PredictionResult(pydantic.BaseModel):
    # model_config  = pydantic.ConfigDict(arbitrary_types_allowed=True)
    class Config:
        arbitrary_types_allowed = True
    image_id: int = None
    bboxes: np.ndarray = None
    scores: np.ndarray = None
    cls: np.ndarray = None
    names: List[str] = None
    annotation_ids: List[int] = None

    def to_coco(self):
        # Convert to coco format
        
        coco_result = []
        for i in range(len(self.bboxes)):
            bbox = self.bboxes[i]
            bbox = box_utils.xyxy2xywh(bbox)
            score = self.scores[i]
            cls = int(self.cls[i])
            rec = {
                'image_id': self.image_id,
                'bbox': bbox,
                'score': score,
                'category_id': cls,
                'iscrowd': 0,
                'area': bbox[2] * bbox[3],
            }
            if self.annotation_ids is not None:
                rec['id'] = int(self.annotation_ids[i])
            else:
                rec['id'] = -1
            coco_result.append(rec)
        return coco_result


class HQTrainerArguments(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(protected_namespaces=())    # 模型配置
    
    # 数据集相关
    data_path: str    # 数据集路径
    image_size: int = 640    # 图像大小
    num_data_workers: int = 0    # 数据加载器线程数
    
    # 训练相关
    num_epoches: int = 100    # 训练轮数
    warmup_epochs: int = 5    # 预热轮数
    batch_size: int = 4    # 批量大小
    gradient_update_interval: int = 1    # 梯度更新间隔
    enable_amp: bool = False    # 是否启用混合精度训练
    max_grad_norm: float = 5.0    # 梯度裁剪阈值
    early_stopping: bool = False    # 是否启用早停
    early_stopping_patience: int = 10   # 早停轮数
    
    # 优化器相关
    lr0: float = 1e-4    # 初始学习率
    lr_min: float = 1e-6    # 最小学习率
    lr_backbone_mult: float = 1    # 主干网络学习率倍数
    
    # 设备相关
    devices: List[int] = [0]    # 设备列表
    
    # 输出相关
    checkpoint_path: str = 'output'    # 检查点路径
    checkpoint_name: str = 'ckpt'    # 检查点名称
    output_path: str = 'output'    # 输出路径
    checkpoint_interval: int = 10    # 检查点间隔
    
    # 模型相关
    model_argument: dict = {}    # 模型参数
    
    # 类别相关
    class_id2names: dict = None    # 类别ID到名称的映射
    eval_class_names: List[str] = None    # 评估类别名称

    find_unused_parameters: bool = False  # DDP 寻找未使用的参数