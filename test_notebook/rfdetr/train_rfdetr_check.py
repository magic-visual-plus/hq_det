from collections import defaultdict
import torch
from hq_det.tools.train_rfdetr import MyTrainer
import hq_det.models.rfdetr.util.misc as utils
from hq_det.models.rfdetr.datasets.coco import CocoDetection as HQCocoDetection
from hq_det.models import rfdetr
from hq_det.models.rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from torch.utils.data import DataLoader, DistributedSampler
from hq_det.common import HQTrainerArguments
from hq_det.models.rfdetr.hq_rfdetr import HQRFDETR
from hq_det.models.rfdetr.engine import train_one_epoch, evaluate
from hq_det.models.rfdetr.util.drop_scheduler import drop_scheduler
from hq_det.models.rfdetr.util.metrics import MetricsPlotSink, MetricsTensorBoardSink, MetricsWandBSink





def cxcywh_norm_to_xyxy_simple(bboxes_cxcywh_norm, img_width, img_height):  
    """简化版本"""  
    bboxes = bboxes_cxcywh_norm.clone()  
    
    # 反归一化  
    bboxes[:, [0, 2]] *= img_width   # cx和w  
    bboxes[:, [1, 3]] *= img_height  # cy和h  
    
    # 转换格式: [cx, cy, w, h] -> [x1, y1, x2, y2]  
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # x1 = cx - w/2  
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # y1 = cy - h/2  
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]      # x2 = x1 + w  
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]      # y2 = y1 + h  
    
    return bboxes  


class MyTrainerCheck(MyTrainer):
    # def __init__(self, args: HQTrainerArguments):
    #     super(MyTrainerCheck, self).__init__(args)

    
    # def ___setup_training_environment(self):
    #     model = HQRFDETR(class_id2names=self.args.class_id2names, **self.args.model_argument)
    #     self.model_args = model.args
    #     args = self.model_args
    #     self.train_config = model.train_config
    #     self.model_config = model.model_config
    #     self.effective_batch_size = args.batch_size * args.grad_accum_steps
    #     effective_batch_size = self.effective_batch_size
    #     super().setup_training_environment()
    #     # self.setup_callbacks()

    #     total_batch_size = effective_batch_size * utils.get_world_size()
    #     num_training_steps_per_epoch = (len(self.dataset_train) + total_batch_size - 1) // total_batch_size
    #     # 初始化调度器字典，用于存储dropout和drop path的调度策略
    #     schedules = {}
    #     # 如果设置了dropout参数大于0，创建dropout调度器
    #     if args.dropout > 0:
    #         # 创建dropout调度器，传入初始dropout值、总训练轮数、每轮训练步数、
    #         # 截止轮数、drop模式和drop调度策略
    #         schedules['do'] = drop_scheduler(
    #             args.dropout, args.epochs, num_training_steps_per_epoch,
    #             args.cutoff_epoch, args.drop_mode, args.drop_schedule)
    #         # 打印dropout调度器的最小值和最大值，保留7位小数
    #         print("Min DO = %.7f, Max DO = %.7f" % (min(schedules['do']), max(schedules['do'])))

    #     # 如果设置了drop path参数大于0，创建drop path调度器
    #     if args.drop_path > 0:
    #         # 创建drop path调度器，参数与dropout调度器类似
    #         schedules['dp'] = drop_scheduler(
    #             args.drop_path, args.epochs, num_training_steps_per_epoch,
    #             args.cutoff_epoch, args.drop_mode, args.drop_schedule)
    #         # 打印drop path调度器的最小值和最大值，保留7位小数
    #         print("Min DP = %.7f, Max DP = %.7f" % (min(schedules['dp']), max(schedules['dp'])))
        
    #     self.schedules = schedules
    #     self.num_training_steps_per_epoch = num_training_steps_per_epoch
    
    # def setup_callbacks(self):
    #     self.callbacks = defaultdict(list)

    #     config = self.train_config
    #     metrics_plot_sink = MetricsPlotSink(output_dir=config.output_dir)
    #     self.callbacks["on_fit_epoch_end"].append(metrics_plot_sink.update)
    #     self.callbacks["on_train_end"].append(metrics_plot_sink.save)

    #     if config.tensorboard:
    #         metrics_tensor_board_sink = MetricsTensorBoardSink(output_dir=config.output_dir)
    #         self.callbacks["on_fit_epoch_end"].append(metrics_tensor_board_sink.update)
    #         self.callbacks["on_train_end"].append(metrics_tensor_board_sink.close)

    #     if config.wandb:
    #         metrics_wandb_sink = MetricsWandBSink(
    #             output_dir=config.output_dir,
    #             project=config.project,
    #             run=config.run,
    #             config=config.model_dump()
    #         )
    #         self.callbacks["on_fit_epoch_end"].append(metrics_wandb_sink.update)
    #         self.callbacks["on_train_end"].append(metrics_wandb_sink.close)

    #     if config.early_stopping:
    #         from hq_det.models.rfdetr.util.early_stopping import EarlyStoppingCallback
    #         early_stopping_callback = EarlyStoppingCallback(
    #             model=self.model,
    #             patience=config.early_stopping_patience,
    #             min_delta=config.early_stopping_min_delta,
    #             use_ema=config.early_stopping_use_ema
    #         )
    #         self.callbacks["on_fit_epoch_end"].append(early_stopping_callback.update)

    def _process_validation_results(self,  all_preds, all_gts, eval_class_ids) -> dict:
        stat = super()._process_validation_results(all_preds, all_gts, eval_class_ids)
        model = self.model
        criterion = model.criterion
        postprocessors = model.postprocessors
        dataloader_val = self.dataloader_val
        base_ds = get_coco_api_from_dataset(dataloader_val)
        test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, dataloader_val, base_ds, self.device, self.model_args)

        return stat
    
    
    # def __setup_datasets_and_transforms(self):
    #     dataset_train = build_dataset(image_set='train', args=self.model_args, resolution=self.model_args.resolution)
    #     dataset_val = build_dataset(image_set='val', args=self.model_args, resolution=self.model_args.resolution)
    #     return dataset_train, dataset_val

    # def __setup_dataloaders(self) -> tuple[
    #     torch.utils.data.DataLoader, 
    #     torch.utils.data.DataLoader
    # ]:
    #     """setup dataloaders"""
    #     data_loader_train = DataLoader(
    #             self.dataset_train, 
    #             batch_size=self.effective_batch_size,
    #             sampler=self.sampler_train,
    #             collate_fn=self.collate_fn, 
    #             num_workers=self.model_args.num_workers
    #         )
    #     data_loader_val = DataLoader(
    #             self.dataset_val, 
    #             batch_size=self.args.batch_size,
    #             sampler=self.sampler_val,
    #             collate_fn=self.collate_fn, 
    #             num_workers=self.model_args.num_workers
    #         )
    #     return data_loader_train, data_loader_val

    # def _collate_fn(self, batch):
    #     batch = utils.collate_fn(batch)

    #     return batch
    #     samples, targets = batch

    #     # new_batch = {}
    #     # new_batch['image_id'] = [b['image_id'] for b in targets]
    #     # new_batch['bboxes_xyxy'] = torch.cat([cxcywh_norm_to_xyxy_simple(b['boxes'], b['size'][0], b['size'][1]) for b in targets], 0)
    #     # new_batch['cls'] = torch.cat([b['labels'] for b in targets], 0)
    #     # batch_idx_list = []
    #     # for i, b in enumerate(targets):
    #     #     batch_idx = torch.full((len(b['labels']),), i, dtype=torch.long)
    #     #     batch_idx_list.append(batch_idx)
    #     # new_batch['batch_idx'] = torch.cat(batch_idx_list, 0)
    #     # new_batch['img'] = samples
    #     # new_batch['targets'] = targets
        
    #     # return new_batch

    # def _train_epoch(self, epoch):
    #     data_loader_train = self.dataloader_train
    #     effective_batch_size = self.effective_batch_size
    #     num_training_steps_per_epoch = self.num_training_steps_per_epoch
    #     model = self.model.model
    #     criterion = self.model.criterion
    #     optimizer = self.optimizer
    #     lr_scheduler = self.scheduler
    #     device = self.device
    #     batch_size = self.args.batch_size
    #     max_norm = self.args.max_grad_norm
    #     ema_m = None
    #     schedules = self.schedules
    #     vit_encoder_num_layers = self.model_args.vit_encoder_num_layers
    #     args = self.model_args
    #     callbacks = self.callbacks
    #     train_stats = train_one_epoch(
    #             model, criterion, lr_scheduler, data_loader_train, optimizer, device, epoch,
    #             effective_batch_size, args.clip_max_norm, ema_m=ema_m, schedules=schedules, 
    #             num_training_steps_per_epoch=num_training_steps_per_epoch,
    #             vit_encoder_num_layers=vit_encoder_num_layers, args=args, callbacks=callbacks)
    #     return train_stats


