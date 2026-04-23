import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.patches as patches
from hq_det.dataset import CocoDetection as HQCocoDetection


class TransformTracker:
    """
    变换跟踪器，用于跟踪transforms中每一步的中间结果
    """
    
    def __init__(self, dataset_root_path):
        """
        Args:
            dataset_root_path: 数据集根路径
        """
        self.dataset_root_path = dataset_root_path
        self._datasets = {}
    
    def __call__(self, dataset_type, target_id, transforms):
        """
        获取指定数据集类型中对应image_id的批次，并跟踪transforms的每一步结果
        
        Args:
            dataset_type: 数据集类型 ('train', 'valid', 'test')
            target_id: 目标image_id
            transforms: 要应用的变换序列
        
        Returns:
            dict: 包含原始数据和transforms每一步结果的字典
        """
        # 懒加载数据集
        if dataset_type not in self._datasets:
            annotation_file = os.path.join(self.dataset_root_path, dataset_type, "_annotations.coco.json")
            image_path = os.path.join(self.dataset_root_path, dataset_type)
            
            self._datasets[dataset_type] = HQCocoDetection(
                image_path, 
                annotation_file, 
                transforms=None
            )
        
        dataset = self._datasets[dataset_type]
        
        # 使用二分法查找
        left, right = 0, len(dataset) - 1
        
        while left <= right:
            mid = (left + right) // 2
            batch = dataset[mid]
            current_id = batch['image_id']
            
            if current_id == target_id:
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
            
            # 应用transform
            if hasattr(transform, '__call__'):
                current_batch = transform(current_batch)
            
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
            bboxes = batch.get('bboxes', [])
            
            # 处理不同类型的图像数据
            if isinstance(img, torch.Tensor):
                # 如果是tensor，转换为numpy
                if img.dim() == 3 and img.shape[0] in [1, 3, 4]:  # (C, H, W)格式
                    img = img.permute(1, 2, 0).cpu().numpy()
                else:
                    img = img.cpu().numpy()
            
            elif isinstance(img, np.ndarray):
                # 如果是numpy数组，确保是uint8格式
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
            
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
            if len(bboxes) > 0:
                h, w = img.shape[:2]
                
                for bbox in bboxes:
                    if isinstance(bbox, torch.Tensor):
                        bbox = bbox.cpu().numpy()
                    
                    # 处理不同格式的边界框
                    if len(bbox) == 4:
                        if bbox.max() <= 1.0:  # 归一化坐标
                            x1, y1, x2, y2 = bbox * [w, h, w, h]
                        else:  # 像素坐标
                            x1, y1, x2, y2 = bbox
                        
                        # 创建矩形
                        rect = patches.Rectangle(
                            (x1, y1), x2 - x1, y2 - y1,
                            linewidth=2, edgecolor='red', facecolor='none'
                        )
                        ax.add_patch(rect)
            
            # 计算图像的mean和std
            if len(img.shape) == 3:
                # 对于彩色图像，计算每个通道的mean和std
                mean_rgb = np.mean(img, axis=(0, 1))
                std_rgb = np.std(img, axis=(0, 1))
                mean_str = f"Mean: [{mean_rgb[0]:.2f}, {mean_rgb[1]:.2f}, {mean_rgb[2]:.2f}]"
                std_str = f"Std: [{std_rgb[0]:.2f}, {std_rgb[1]:.2f}, {std_rgb[2]:.2f}]"
            else:
                # 对于灰度图像
                mean_val = np.mean(img)
                std_val = np.std(img)
                mean_str = f"Mean: {mean_val:.2f}"
                std_str = f"Std: {std_val:.2f}"
            
            # 添加图像信息，包括mean和std
            info_text = f"Shape: {img.shape}\nBoxes: {len(bboxes)}\n{mean_str}\n{std_str}"
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
