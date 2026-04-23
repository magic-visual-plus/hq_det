from loguru import logger
import torch
from torch.utils.data import DataLoader, DistributedSampler
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from hq_det.models.rfdetr.detr import RFDETRBase
from hq_det.models.rfdetr.main import build_dataset, populate_args
from hq_det.models.rfdetr.config import TrainConfig, RFDETRBaseConfig
import hq_det.models.rfdetr.util.misc as utils
from hq_det.models.rfdetr.datasets import build_dataset, get_coco_api_from_dataset

from hq_det.models.rfdetr.datasets.coco import CocoDetection as RFDETRCocoDetection


def create_dataloaders(
    dataset_dir: str,
    pretrain_weights: str = "/root/autodl-tmp/model/rfdetr/rf-detr-base.pth",
    square_resize_div_64: bool = False,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    num_workers: int = 2,
    distributed: bool = False
):
    """
    创建训练和验证数据加载器
    
    Args:
        dataset_dir: 数据集目录路径
        pretrain_weights: 预训练权重路径
        square_resize_div_64: 是否进行64整除的正方形调整
        batch_size: 批次大小
        grad_accum_steps: 梯度累积步数
        num_workers: 数据加载器工作进程数
        distributed: 是否使用分布式训练
    
    Returns:
        tuple: (data_loader_train, data_loader_val)
    """
    train_config = TrainConfig(
        dataset_dir=dataset_dir,
        square_resize_div_64=square_resize_div_64,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        num_workers=num_workers
    )

    model_config = RFDETRBaseConfig(
        pretrain_weights=pretrain_weights,
    )

    args = populate_args(**{**model_config.dict(), **train_config.dict()})

    dataset_train = build_dataset(image_set='train', args=args, resolution=args.resolution)
    dataset_val = build_dataset(image_set='val', args=args, resolution=args.resolution)

    if distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    effective_batch_size = args.batch_size * args.grad_accum_steps

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, effective_batch_size, drop_last=True)
    data_loader_train = DataLoader(
        dataset_train, 
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn, 
        num_workers=args.num_workers
    )

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                drop_last=False, collate_fn=utils.collate_fn, 
                                num_workers=args.num_workers)
    
    return data_loader_train, data_loader_val


def flatten_transforms(transforms):
    """递归展平transforms，消除Compose，返回所有transforms的列表"""
    result = []
    
    if hasattr(transforms, 'transforms'):
        # 如果是Compose类型，递归处理其内部的transforms
        for t in transforms.transforms:
            result.extend(flatten_transforms(t))
    else:
        # 如果是单个transform，直接添加
        result.append(transforms)
    
    return result


class RFDETRTransformTracker:
    """
    RFDETR 变换跟踪器，用于跟踪transforms中每一步的中间结果
    """
    
    def __init__(self, dataset_dir):
        """
        Args:
            dataset_dir: 数据集目录路径
            pretrain_weights: 预训练权重路径
            square_resize_div_64: 是否进行64整除的正方形调整
            resolution: 图像分辨率
        """
        self.dataset_dir = dataset_dir
        self._datasets = {}
        self._args = None
    
    def __call__(self, dataset_type, target_id, transforms):
        """
        获取指定数据集类型中对应image_id的批次，并跟踪transforms的每一步结果
        
        Args:
            dataset_type: 数据集类型 ('train', 'val')
            target_id: 目标image_id
            transforms: 要应用的变换序列
        
        Returns:
            dict: 包含原始数据和transforms每一步结果的字典
        """
        dataset = RFDETRCocoDetection(
            f"{self.dataset_dir}/{dataset_type}",
            f"{self.dataset_dir}/{dataset_type}/_annotations.coco.json",
            transforms=None
        )
        
        # 使用二分法查找
        left, right = 0, len(dataset) - 1
        
        while left <= right:
            mid = (left + right) // 2
            img, target = dataset[mid]
            current_id = target['image_id'].item() if isinstance(target['image_id'], torch.Tensor) else target['image_id']
            
            if current_id == target_id:
                # 将 (img, target) 转换为 {'img': img, **target} 格式
                batch = {'img': img, **target}
                return self._track_transforms(batch, transforms)
            elif current_id < target_id:
                left = mid + 1
            else:
                right = mid - 1
        
        return None
    
    def _track_transforms(self, original_batch, transforms):
        """
        跟踪transforms的每一步结果
        
        Args:
            original_batch: 原始批次数据
            transforms: 要应用的变换序列
        
        Returns:
            dict: 包含原始数据和transforms每一步结果的字典
        """
        results = {
            'original': original_batch,
            'transform_steps': []
        }
        current_batch = original_batch.copy()
        
        for i, transform in enumerate(transforms):
            step_result = {
                'step': i,
                'transform_name': transform.__class__.__name__,
                'batch_before': current_batch.copy()
            }
            
            # 应用transform - RFDETR transforms 期望 (img, target) 格式
            if hasattr(transform, '__call__'):
                img = current_batch['img']
                target = {k: v for k, v in current_batch.items() if k != 'img'}
                
                # 应用transform
                img, target = transform(img, target)
                
                # 重新组合为batch格式
                current_batch = {'img': img, **target}
            
            step_result['batch_after'] = current_batch.copy()
            results['transform_steps'].append(step_result)
        
        # 添加最终结果
        results['final'] = {
            'batch': current_batch.copy()
        }
        
        return results
    
    def visualize_transforms(self, transform_results, save_path=None, max_cols=3):
        """
        可视化transforms每一步的结果
        
        Args:
            transform_results: _track_transforms返回的结果
            save_path: 保存路径，如果为None则显示图像
            max_cols: 每行最大显示的图像数量
        """
        original_batch = transform_results['original']
        transform_steps = transform_results['transform_steps']
        
        # 计算子图布局
        total_images = 1 + len(transform_steps)  # 原始图像 + 每一步的结果
        n_cols = min(max_cols, total_images)
        n_rows = (total_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # 绘制原始图像
        self._plot_image_with_boxes(original_batch, axes[0, 0], "Original")
        
        # 绘制每一步的结果
        for i, step in enumerate(transform_steps, 1):
            row = i // n_cols
            col = i % n_cols
            title = f"Step {step['step']}: {step['transform_name']}"
            self._plot_image_with_boxes(step['batch_after'], axes[row, col], title)
        
        # 隐藏多余的子图
        for i in range(total_images, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _plot_image_with_boxes(self, batch, ax, title):
        """
        在给定的axes上绘制图像和边界框
        
        Args:
            batch: 包含图像和边界框的批次数据
            ax: matplotlib axes对象
            title: 图像标题
        """
        try:
            # 获取图像数据
            img = batch['img']
            boxes = batch.get('boxes', [])
            
            # 保存原始图像数据用于统计
            original_img = img
            
            # 处理不同类型的图像数据
            if isinstance(img, torch.Tensor):
                # 如果是tensor，转换为numpy
                if img.dim() == 3 and img.shape[0] in [1, 3, 4]:  # (C, H, W)格式
                    img = img.permute(1, 2, 0).cpu().numpy()
                else:
                    img = img.cpu().numpy()
                
                # 处理归一化的图像数据
                if img.dtype == np.float32 or img.dtype == np.float64:
                    # 转换为uint8格式用于显示
                    img = (img).astype(np.uint8)
            
            elif isinstance(img, np.ndarray):
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = img.astype(np.uint8)
            
            elif hasattr(img, 'size'):  # PIL Image对象
                # 将PIL Image转换为numpy数组
                img = np.array(img)
                if len(img.shape) == 3 and img.shape[2] == 3:
                    # 确保是RGB格式
                    if img.dtype == np.uint8:
                        import cv2
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 确保图像是RGB格式
            if len(img.shape) == 3 and img.shape[2] == 3:
                # 如果是BGR格式，转换为RGB
                if img.dtype == np.uint8:
                    import cv2
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 绘制图像
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            
            # 绘制边界框
            if len(boxes) > 0:
                h, w = img.shape[:2]
                
                for box in boxes:
                    if isinstance(box, torch.Tensor):
                        box = box.cpu().numpy()
                    
                    # 处理不同格式的边界框
                    if len(box) == 4:
                        if box.max() <= 1.0:  # 归一化坐标
                            x1, y1, x2, y2 = box * [w, h, w, h]
                        else:  # 像素坐标
                            x1, y1, x2, y2 = box
                        
                        # 创建矩形
                        rect = patches.Rectangle(
                            (x1, y1), x2 - x1, y2 - y1,
                            linewidth=2, edgecolor='red', facecolor='none'
                        )
                        ax.add_patch(rect)
            
            # 使用原始图像数据计算mean和std
            if isinstance(original_img, torch.Tensor):
                # 如果是tensor，转换为numpy用于统计
                if original_img.dim() == 3 and original_img.shape[0] in [1, 3, 4]:  # (C, H, W)格式
                    original_img_np = original_img.permute(1, 2, 0).cpu().numpy()
                else:
                    original_img_np = original_img.cpu().numpy()
            elif isinstance(original_img, np.ndarray):
                original_img_np = original_img
            elif hasattr(original_img, 'size'):  # PIL Image对象
                original_img_np = np.array(original_img)
            else:
                original_img_np = original_img
            
            # 计算图像的mean和std
            if len(original_img_np.shape) == 3:
                # 对于彩色图像，计算每个通道的mean和std
                mean_rgb = np.mean(original_img_np, axis=(0, 1))
                std_rgb = np.std(original_img_np, axis=(0, 1))
                mean_str = f"Mean: [{mean_rgb[0]:.2f}, {mean_rgb[1]:.2f}, {mean_rgb[2]:.2f}]"
                std_str = f"Std: [{std_rgb[0]:.2f}, {std_rgb[1]:.2f}, {std_rgb[2]:.2f}]"
            else:
                # 对于灰度图像
                mean_val = np.mean(original_img_np)
                std_val = np.std(original_img_np)
                mean_str = f"Mean: {mean_val:.2f}"
                std_str = f"Std: {std_val:.2f}"
            
            # 添加图像信息，包括mean和std
            info_text = f"Shape: {original_img_np.shape}\nBoxes: {len(boxes)}\n{mean_str}\n{std_str}"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
        except Exception as e:
            # 如果绘制失败，显示错误信息
            ax.text(0.5, 0.5, f"Error: {str(e)}", 
                    transform=ax.transAxes, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
            ax.set_title(f"{title} (Error)", fontsize=10, color='red')
            ax.axis('off')
            print(f"绘制图像时出错 {title}: {str(e)}")
