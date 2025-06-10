from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Iterable
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

@dataclass
class ExperimentInfo:
    name: str
    csv_path: str
    df: Optional[pd.DataFrame] = None
    log_path: Optional[str] = None
    framework: Optional[str] = 'hq_det'  # 实验使用的代码框架，如 hq_det, mmdet 等

# 定义COCO评估指标的正则表达式模式
COCO_METRICS_PATTERNS = {
    'AP': {
        'mAP': r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([\d\.]+)",
        'AP50': r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([\d\.]+)",
        'AP75': r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.75\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([\d\.]+)",
        'AP_small': r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*small\s*\|\s*maxDets=100\s*\]\s*=\s*([\d\.]+)",
        'AP_medium': r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*medium\s*\|\s*maxDets=100\s*\]\s*=\s*([\d\.]+)",
        'AP_large': r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*large\s*\|\s*maxDets=100\s*\]\s*=\s*([\d\.]+)"
    },
    'AR': {
        'AR_1': r"Average Recall\s+\(AR\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*1\s*\]\s*=\s*([\d\.]+)",
        'AR_10': r"Average Recall\s+\(AR\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*10\s*\]\s*=\s*([\d\.]+)",
        'AR_100': r"Average Recall\s+\(AR\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([\d\.]+)",
        'AR_small': r"Average Recall\s+\(AR\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*small\s*\|\s*maxDets=100\s*\]\s*=\s*([\d\.]+)",
        'AR_medium': r"Average Recall\s+\(AR\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*medium\s*\|\s*maxDets=100\s*\]\s*=\s*([\d\.]+)",
        'AR_large': r"Average Recall\s+\(AR\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*large\s*\|\s*maxDets=100\s*\]\s*=\s*([\d\.]+)"
    }
}

class ComparerBase:

    def __init__(self, experiments: List[ExperimentInfo]):
        """
        初始化比较器基类
        
        Args:
            experiments (List[ExperimentInfo]): 实验信息对象列表
        """
        self.experiments = self.load_experiment_results(experiments)


    def _extract_metrics_from_log(self, log_content: str) -> Dict[str, float]:
        """
        从日志内容中提取COCO评估指标
        
        Args:
            log_content (str): 日志文件内容
            
        Returns:
            Dict[str, float]: 提取的指标字典，键为指标名称，值为指标值
        """
        metrics = {}
        
        # 遍历所有指标类型（AP和AR）
        for metric_type, patterns in COCO_METRICS_PATTERNS.items():
            # 遍历每种类型的具体指标
            for metric_name, pattern in patterns.items():
                for line in log_content:
                    match = re.search(pattern, line, re.MULTILINE)
                    if match:
                        metrics.setdefault(metric_name, []).append(float(match.group(1)))
                    
        return metrics

    def _update_experiment_metrics(self, exp: ExperimentInfo, metrics: Dict[str, float]):
        """
        更新实验数据框中的指标
        
        Args:
            exp (ExperimentInfo): 实验信息对象
            metrics (Dict[str, float]): 要添加的指标字典
        """
        for metric_name, value in metrics.items():
            if metric_name not in exp.df.columns:
                exp.df[metric_name] = value

    def load_experiment_results(self, experiments: List[ExperimentInfo]) -> List[ExperimentInfo]:
        """
        加载和处理实验结果
        
        Args:
            experiments (List[ExperimentInfo]): 实验信息对象列表
            
        Returns:
            List[ExperimentInfo]: 更新后的实验信息对象列表，包含加载的DataFrames和指标
        """
        for exp in experiments:
            try:
                # 尝试加载CSV数据
                if exp.csv_path:
                    exp.df = pd.read_csv(exp.csv_path)
                else:
                    exp.df = pd.DataFrame()
                
                # 如果存在日志文件，提取并添加COCO指标
                if exp.log_path:
                    with open(exp.log_path, 'r', encoding='utf-8') as f:
                        log_content = f.readlines()
                    metrics = self._extract_metrics_from_log(log_content)
                    self._update_experiment_metrics(exp, metrics)
                    
            except Exception as e:
                print(f"加载 {exp.name} 时出错: {str(e)}")
                exp.df = pd.DataFrame()
                
        return experiments

    def _get_loss_types(self):
        """
        获取损失类型列名和显示名称
        
        Returns:
            List[Tuple[str, str, str]]: 包含(train_loss, val_loss, loss_name)元组的列表
        """
        return [
            ('train/box_loss', 'val/box_loss', 'Box Loss'),
            ('train/giou_loss', 'val/giou_loss', 'GIoU Loss'),
            ('train/cls_loss', 'val/cls_loss', 'Classification Loss')
        ]

    def _get_metric_configs(self):
        """
        获取性能指标列名和显示名称
        
        Returns:
            List[Tuple[str, str]]: 包含(metric_key, display_name)元组的列表
        """
        return [
            ('mAP', 'mAP'),
            ('precision', 'Precision'),
            ('recall', 'Recall'),
            ('f1_score', 'F1 Score'),
            ('fnr', 'False Negative Rate'),
            ('confidence', 'Confidence')
        ]

    def _plot_metric_subplot(self, ax, metric: str, title: str, experiments: List[ExperimentInfo]):
        """
        为多个实验绘制单个指标子图
        
        Args:
            ax: Matplotlib轴对象
            metric (str): DataFrame中的指标列名
            title (str): 图表标题
            experiments (List[ExperimentInfo]): 要绘制的实验列表
        """
        for exp in experiments:
            if exp.df is not None and metric in exp.df.columns:
                ax.plot(exp.df[metric], label=exp.name)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    def _create_subplots(self, n_rows: int, n_cols: int, figsize: Tuple[int, int] = (15, 10)):
        """
        创建带有子图的图形
        
        Args:
            n_rows (int): 子图网格的行数
            n_cols (int): 子图网格的列数
            figsize (Tuple[int, int]): 图形大小 (宽度, 高度)
            
        Returns:
            Tuple[plt.Figure, List[plt.Axes]]: 图形和轴对象列表
        """
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        return fig, axes.flatten()

    def plot_best_metrics_comparison(self, figsize: Tuple[int, int] = (15, 8), show_value: bool = True, return_fig: bool = False, metrics: Optional[List[Tuple[str, str]]] = None):
        """
        绘制不同实验的最佳性能指标对比直方图
        
        Args:
            figsize (Tuple[int, int]): 图形大小 (宽度, 高度)
            show_value (bool): 是否在柱状图顶部显示数值
            return_fig (bool): 是否返回图形对象而不是显示
            metrics (Optional[List[Tuple[str, str]]]): 要对比的指标列表，每个元素为(metric_key, display_name)元组。
                                                      如果为None，则使用默认指标配置
            
        Returns:
            Optional[Tuple[plt.Figure, plt.Axes]]: 如果return_fig为True，返回图形和轴对象
        """
        def metric_config(metrics):
            if metrics is None:
                return self._get_metric_configs()
            elif  isinstance(metrics, str):
                return [(metrics, metrics)]
            else:
                metrics = list(metrics)
                for i, metric in enumerate(metrics):
                    if isinstance(metric, str):
                        metrics[i] = (metric, metric)
                    elif isinstance(metric, Iterable):
                        assert len(metric) == 2, ""
                    else:
                        ValueError
                return metrics

        # 使用传入的指标配置或默认配置
        metrics = metric_config(metrics)
        n_metrics = len(metrics)
        
        # 创建图形和轴
        fig, ax = plt.subplots(figsize=figsize)
        
        # 设置柱状图的位置
        x = np.arange(n_metrics)
        width = 0.8 / len(self.experiments)
        
        # 为每个实验收集最佳指标值
        best_values = []
        for exp in self.experiments:
            if exp.df is not None:
                exp_best = []
                for metric_key, _ in metrics:
                    if metric_key in exp.df.columns:
                        # 对于fnr，取最小值；对于其他指标，取最大值
                        if metric_key == 'fnr':
                            best_value = exp.df[metric_key].min()
                        else:
                            best_value = exp.df[metric_key].max()
                        exp_best.append(best_value)
                    else:
                        exp_best.append(0)
                best_values.append(exp_best)
        
        # 绘制柱状图
        for i, exp in enumerate(self.experiments):
            if exp.df is not None:
                offset = (i - len(self.experiments)/2 + 0.5) * width
                bars = ax.bar(x + offset, best_values[i], width, label=exp.name)
                
                # 在柱状图顶部显示数值
                if show_value:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.3f}',
                                ha='center', va='bottom')
        
        # 设置图表属性
        ax.set_ylabel('Value')
        ax.set_title('Best Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m[1] for m in metrics], rotation=45)
        ax.legend()
        ax.grid(True, axis='y')
        
        # 调整布局
        plt.tight_layout()
        if return_fig:
            return fig, ax
        else:
            plt.show()