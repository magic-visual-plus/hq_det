import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List
from .base import ComparerBase, ExperimentInfo

class DiffModelParamComparer(ComparerBase):
    def plot_loss_comparison(self, figsize=(15, 10)):
        """
        绘制同一模型不同参数配置下的损失曲线对比
        
        Args:
            figsize (tuple): 图像大小 (宽度, 高度)
        """
        loss_types = self._get_loss_types()
        fig, axes = self._create_subplots(3, 2, figsize)
        
        for idx, (train_loss, val_loss, loss_name) in enumerate(loss_types):
            # 绘制训练损失
            self._plot_metric_subplot(axes[2*idx], train_loss, f'Training {loss_name}', self.experiments)
            # 绘制验证损失
            self._plot_metric_subplot(axes[2*idx+1], val_loss, f'Validation {loss_name}', self.experiments)
        
        plt.tight_layout()
        plt.show()

    def plot_metrics_comparison(self, figsize=(15, 10)):
        """
        绘制同一模型不同参数配置下的性能指标对比
        
        Args:
            figsize (tuple): 图像大小 (宽度, 高度)
        """
        metric_configs = self._get_metric_configs()
        fig, axes = self._create_subplots(3, 2, figsize)
        
        for idx, (metric_key, display_name) in enumerate(metric_configs):
            self._plot_metric_subplot(axes[idx], metric_key, display_name, self.experiments)
        
        plt.tight_layout()
        plt.show()


class DiffModelTypeComparer(ComparerBase):
    def __init__(self, experiments: List[ExperimentInfo], training_times: Union[float, List[float]]):
        """
        初始化不同模型比较器
        
        Args:
            experiments (List[ExperimentInfo]): 实验信息列表
            training_times (Union[float, List[float]]): 训练总时长(小时)，可以是单个值或与实验数量相同的列表
        """
        super().__init__(experiments)
        if isinstance(training_times, (int, float)):
            self.training_times = [training_times] * len(experiments)
        else:
            if len(training_times) != len(experiments):
                raise ValueError("训练时长列表长度必须与实验数量相同")
            self.training_times = training_times
        
    def _convert_to_time_axis(self, df: pd.DataFrame, total_time: float) -> pd.Series:
        """
        将epoch转换为时间轴
        
        Args:
            df (pd.DataFrame): 实验数据
            total_time (float): 总训练时长(小时)
            
        Returns:
            pd.Series: 时间轴数据
        """
        n_epochs = len(df)
        return pd.Series(np.linspace(0, total_time, n_epochs))
    
    def _plot_time_series(self, ax, data: pd.Series, time_axis: pd.Series, label: str):
        """
        绘制时间序列数据
        
        Args:
            ax: Matplotlib轴对象
            data (pd.Series): 要绘制的数据
            time_axis (pd.Series): 时间轴数据
            label (str): 图例标签
        """
        ax.plot(time_axis, data, label=label)
        ax.set_xlabel('Training Time (hours)')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        
    def plot_loss_comparison(self, figsize=(15, 10)):
        """
        绘制不同模型不同参数配置下的损失曲线对比，x轴为训练时间
        """
        loss_types = self._get_loss_types()
        fig, axes = self._create_subplots(3, 2, figsize)
        
        for idx, (train_loss, val_loss, loss_name) in enumerate(loss_types):
            # 绘制训练损失
            ax_train = axes[2*idx]
            for i, exp in enumerate(self.experiments):
                if exp.df is not None and train_loss in exp.df.columns:
                    time_axis = self._convert_to_time_axis(exp.df, self.training_times[i])
                    self._plot_time_series(ax_train, exp.df[train_loss], time_axis, exp.name)
            ax_train.set_title(f'Training {loss_name}')
            
            # 绘制验证损失
            ax_val = axes[2*idx+1]
            for i, exp in enumerate(self.experiments):
                if exp.df is not None and val_loss in exp.df.columns:
                    time_axis = self._convert_to_time_axis(exp.df, self.training_times[i])
                    self._plot_time_series(ax_val, exp.df[val_loss], time_axis, exp.name)
            ax_val.set_title(f'Validation {loss_name}')
        
        plt.tight_layout()
        plt.show()
        
    def plot_metrics_comparison(self, figsize=(15, 10)):
        """
        绘制不同模型不同参数配置下的性能指标对比，x轴为训练时间
        """
        metric_configs = self._get_metric_configs()
        fig, axes = self._create_subplots(3, 2, figsize)
        
        for idx, (metric_key, display_name) in enumerate(metric_configs):
            ax = axes[idx]
            for i, exp in enumerate(self.experiments):
                if exp.df is not None and metric_key in exp.df.columns:
                    time_axis = self._convert_to_time_axis(exp.df, self.training_times[i])
                    self._plot_time_series(ax, exp.df[metric_key], time_axis, exp.name)
            ax.set_title(display_name)
        
        plt.tight_layout()
        plt.show()