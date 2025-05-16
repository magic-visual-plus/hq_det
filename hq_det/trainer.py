import os
import time
from . import torch_utils
import torch.utils.data
import pydantic
import loguru
import numpy as np
from . import augment
from . import evaluate
from . import box_utils
from .common import PredictionResult
from tqdm import tqdm
from .models.base import HQModel
import torch
from common import HQTrainerArguments


def add_stats(info1, info2):
    for k, v in info2.items():
        if k not in info1:
            info1[k] = v
        else:
            info1[k] += v
            pass
        pass
    return info1


def divide_stats(info, n):
    for k, v in info.items():
        info[k] = v / n
        pass
    return info


def format_stats(info):
    parts = []
    for k, v in info.items():
        if isinstance(v, float):
            parts.append(f'{k}: {v:.2f}')
        elif isinstance(v, int):
            parts.append(f'{k}: {v}')
        else:
            parts.append(f'{k}: {v}')
            pass
        pass
    return ','.join(parts)


def extract_ground_truth(batch_data):
    batch_size = len(batch_data['image_id'])
    gt_boxes = batch_data['bboxes_xyxy']
    gt_cls = batch_data['cls']
    image_ids = batch_data['image_id']

    gt_records = []
    for i in range(batch_size):
        image_id = image_ids[i]
        mask = batch_data['batch_idx'] == i
        if mask.sum() == 0:
            gt_bboxes = np.zeros((0, 4))
            gt_cls = np.zeros((0,))
            pass
        else:
            # add gt
            gt_bboxes = batch_data['bboxes_xyxy'][mask].cpu().numpy()
            gt_cls = batch_data['cls'][mask].cpu().numpy()
            pass

        record =  PredictionResult(
            image_id=image_id,
            bboxes=gt_bboxes,
            scores=np.ones(gt_bboxes.shape[0]),
            cls=gt_cls,
        )
        gt_records.append(record)
        pass


    return gt_records


class HQTrainer:
    def __init__(self, args: HQTrainerArguments):
        self.args = args
        self.logger = loguru.logger
        self.results_file = os.path.join(args.output_path, 'results.csv')

    def build_dataset(self, train_transforms, val_transforms):
        pass

    def build_model(self):
        pass

    def collate_fn(self, batch):
        # print(batch)
        pass

    def build_transforms(self, aug=True):
        transforms = []

        transforms.append(augment.ToNumpy())

        if aug:
            transforms.append(augment.RandomHorizontalFlip())
            transforms.append(augment.RandomVerticalFlip())
            transforms.append(augment.RandomGrayScale())
            transforms.append(augment.RandomShuffleChannel())
            transforms.append(augment.RandomRotate90(p=0.1))
            transforms.append(augment.RandomRotate(p=0.1))
            transforms.append(augment.RandomAffine(p=0.1))
            transforms.append(augment.RandomPerspective(p=0.1))
            transforms.append(augment.RandomNoise(p=0.1))
            transforms.append(augment.RandomBrightness(p=0.1))
            transforms.append(augment.RandomCrop(p=0.1))
            transforms.append(augment.RandomResize(p=0.1))
            pass
        else:
            pass

        transforms.append(augment.Resize(max_size=self.args.image_size))
        transforms.append(augment.FilterSmallBox())
        transforms.append(augment.Format())

        return augment.Compose(transforms)
        pass

    def build_optimizer(self, model: HQModel):
        param_dict = model.get_param_dict(self.args)
        return torch.optim.AdamW(param_dict, lr=self.args.lr0)

    def get_lr_multi(self, iepoch):
        lr0 = self.args.lr0
        lr_min = self.args.lr_min
        warmup_epochs = self.args.warmup_epochs
        num_epoches = self.args.num_epoches

        if iepoch < warmup_epochs:
            lr = lr0 * (iepoch + 1) / warmup_epochs
        else:
            lr = lr0 * (1.0 - min(iepoch, num_epoches) / num_epoches)
            pass
        
        return max(lr, lr_min)
        pass

    def build_scheduler(self, optimizer):
        # return torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=self.get_lr_multi,
        )
    

    def run(self, ):
        self.logger.info(self.args)
        device = self.args.device
        num_epoches = self.args.num_epoches
        warmup_epochs = self.args.warmup_epochs
        batch_size = self.args.batch_size
        num_data_workers = self.args.num_data_workers
        lr0 = self.args.lr0
        lr_min = self.args.lr_min

        trainsforms_train = self.build_transforms(aug=True)
        trainsforms_val = self.build_transforms(aug=False)
        dataset_train, dataset_val = self.build_dataset(
            trainsforms_train, trainsforms_val)

        if self.args.class_id2names is None:
            self.args.class_id2names = dataset_train.class_id2names
            pass

        if self.args.eval_class_names is None:
            eval_class_ids = list(dataset_train.class_id2names.keys())
        else:
            class_names2id = {v: k for k, v in self.args.class_id2names.items()}
            eval_class_ids = [
                class_names2id[class_name] for class_name in self.args.eval_class_names if class_name in class_names2id]
            pass

        if len(eval_class_ids) == 0:
            # cannot find any class, use all class ids to evaluate
            eval_class_ids = list(dataset_train.class_id2names.keys())
            pass

        model = self.build_model()

        dataloader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_data_workers, collate_fn=self.collate_fn)
        dataloader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_data_workers, collate_fn=self.collate_fn)

        optimizer = self.build_optimizer(model)
        scheduler = self.build_scheduler(optimizer)
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.enable_amp)

        os.makedirs(self.args.checkpoint_path, exist_ok=True)

        model.to(device)
        train_info = dict()
        
        start_time = time.time()
        for i_epoch in range(num_epoches + warmup_epochs):
            # Training process
            train_losses = []
            epoch_start_time = time.time()

            if i_epoch < warmup_epochs:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(lr0 * (i_epoch + 1) / warmup_epochs, lr_min)
                    pass
                pass

            model.train()
            bar_train = tqdm(dataloader_train, desc=f"Train Epoch[{i_epoch}/{num_epoches + warmup_epochs - 1}]")
            train_start_time = time.time()
            for i_batch, batch_data in enumerate(bar_train):
                # print(batch_data['bboxes'])
                batch_data = torch_utils.batch_to_device(batch_data, device)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.enable_amp):
                    # Forward pass
                    forward_result = model(batch_data)
                    # Compute loss
                    # forward_result = torch_utils.nan_to_num(forward_result)
                    loss, info = model.compute_loss(batch_data, forward_result)
                    # calculate averge iou
                    pass
                
                train_losses.append(loss.item())
                bar_train.set_postfix(
                    **info
                )
                train_info = add_stats(train_info, info)
                # Backward pass

                scaler.scale(loss / self.args.gradient_update_interval).backward()
                if i_batch % self.args.gradient_update_interval == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    pass
                pass
            
            train_time = time.time() - train_start_time
            train_hours, remainder = divmod(train_time, 3600)
            train_mins, train_secs = divmod(remainder, 60)
            
            # Validation process
            model.eval()
            val_losses = []
            val_info = dict()
            gt_records = []
            pred_records = []
            num_images = 0
            val_start_time = time.time()
            bar_val = tqdm(dataloader_val, desc=f"Valid Epoch[{i_epoch}/{num_epoches + warmup_epochs - 1}]")
            for i_batch, batch_data in enumerate(bar_val):
                batch_data = torch_utils.batch_to_device(batch_data, device)

                with torch.no_grad():
                    forward_result = model(batch_data)
                    # Compute loss
                    loss, info_ = model.compute_loss(batch_data, forward_result)
                    preds = model.postprocess(forward_result, batch_data)

                    for pred, image_id in zip(preds, batch_data['image_id']):
                        pred.image_id = image_id
                        pass

                    val_info = add_stats(val_info, info_)
                    val_losses.append(loss.item())
                    # calculate averge iou
                    pass
                
                bar_val.set_postfix(**info_)
                pred_records.extend(preds)
                gt_records.extend(extract_ground_truth(batch_data))
                pass
            
            val_time = time.time() - val_start_time
            val_hours, remainder = divmod(val_time, 3600)
            val_mins, val_secs = divmod(remainder, 60)
            
            epoch_time = time.time() - epoch_start_time
            epoch_hours, remainder = divmod(epoch_time, 3600)
            epoch_mins, epoch_secs = divmod(remainder, 60)
            
            train_info = divide_stats(train_info, len(dataloader_train))
            val_info = divide_stats(val_info, len(dataloader_val))

            # Evaluate the model
            # stat = evaluate.eval_detection_result(
            #     gt_records, pred_records, model.get_class_names())

            stat = evaluate.eval_detection_result_by_class_id(
                gt_records, pred_records, eval_class_ids)
            
            for loss_name in ['box', 'cls', 'giou']:
                stat[f'train/{loss_name}_loss'] = train_info[f'{loss_name}']
                stat[f'val/{loss_name}_loss'] = val_info[f'{loss_name}']
                pass
            
            self.save_epoch_result(i_epoch, stat, self.args.output_path)
            self.logger.info(
                f'Epoch {i_epoch}, lr: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses)}, valid loss: {np.mean(val_losses)}, '
                f'{format_stats(val_info)}'
            )
            self.logger.info(
                f'Elapsed Time: Train {int(train_hours):02d}:{int(train_mins):02d}:{int(train_secs):02d} | '
                f'Valid {int(val_hours):02d}:{int(val_mins):02d}:{int(val_secs):02d} | '
                f'Epoch {int(epoch_hours):02d}:{int(epoch_mins):02d}:{int(epoch_secs):02d}'
            )

            if i_epoch >= warmup_epochs:
                scheduler.step()
            # Save checkpoint
            checkpoint_path = os.path.join(self.args.checkpoint_path, 'ckpt')
            model.save(checkpoint_path)

        total_time = time.time() - start_time
        total_hours, remainder = divmod(total_time, 3600)
        total_mins, total_secs = divmod(remainder, 60)
        start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
        end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        self.logger.info(f'Start Time: {start_time_str}, End Time: {end_time_str}')
        self.logger.info(f'Total Time: {int(total_hours):02d}:{int(total_mins):02d}:{int(total_secs):02d}') 


    
    def save_epoch_result(self, iepoch, stat, output_path):
        header = ['mAP', 'precision', 'recall', 'f1_score', 'fnr', 'confidence', 'train/box_loss', 'train/cls_loss', 'train/giou_loss', 'val/box_loss', 'val/cls_loss', 'val/giou_loss']
        
        if iepoch == 0:
            # add header
            with open(self.results_file, 'w') as f:
                f.write(','.join(header) + '\n')

        with open(self.results_file, 'a') as f:
            values = [stat[colname] for colname in header if colname in stat]
            f.write(','.join([str(v) for v in values]) + '\n')
            pass

        # save curve
        plots_path = os.path.join(output_path, 'plots', f'epoch{iepoch}')
        os.makedirs(plots_path, exist_ok=True)
        curve_file = os.path.join(plots_path, 'pr_curve.csv')
        
        precisions = stat['precisions']
        recalls = stat['recalls']

        with open(curve_file, 'w') as f:
            f.write('px,all\n')

            for i in range(len(precisions)):
                f.write(f'{recalls[i]},{precisions[i]}\n')