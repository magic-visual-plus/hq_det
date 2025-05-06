import os
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


class HQTrainerArguments(pydantic.BaseModel):
    model_config  = pydantic.ConfigDict(protected_namespaces=())
    data_path: str
    num_epoches: int = 100
    warmup_epochs: int = 5
    num_data_workers: int = 0
    lr0: float = 1e-4
    lr_min: float = 1e-6
    batch_size: int = 4
    device: str = 'cuda:0'
    checkpoint_path: str = 'checkpoints'
    checkpoint_interval: int = 10
    model_argument: dict = {}
    image_size: int = 640
    
    class_id2names: dict = None
    pass



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
        pass

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
            # transforms.append(augment.RandomRotate())
            # transforms.append(augment.RandomAffine())
            # transforms.append(augment.RandomPerspective())
            # transforms.append(augment.RandomNoise())
            # transforms.append(augment.RandomBrightness())
            # transforms.append(augment.RandomCrop())
            # transforms.append(augment.RandomResize())
            pass
        else:
            pass

        transforms.append(augment.Resize(max_size=self.args.image_size))
        transforms.append(augment.FilterSmallBox())
        transforms.append(augment.Format())

        return augment.Compose(transforms)
        pass

    def build_optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=self.args.lr0)

    def build_scheduler(self, optimizer):
        # return torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0 - min(epoch, self.args.num_epoches) / self.args.num_epoches
        )
    

    def run(self, ):
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

        model = self.build_model()

        dataloader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_data_workers, collate_fn=self.collate_fn)
        dataloader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_data_workers, collate_fn=self.collate_fn)

        optimizer = self.build_optimizer(model)
        scheduler = self.build_scheduler(optimizer)

        os.makedirs(self.args.checkpoint_path, exist_ok=True)

        model.to(device)
        for i_epoch in range(num_epoches + warmup_epochs):
            # Training process
            train_losses = []

            if i_epoch < warmup_epochs:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(lr0 * (i_epoch + 1) / warmup_epochs, lr_min)
                    pass
                pass

            model.train()
            bar = tqdm(dataloader_train)
            for i_batch, batch_data in enumerate(bar):
                # print(batch_data['bboxes'])
                batch_data = torch_utils.batch_to_device(batch_data, device)
                
                forward_result = model(batch_data)
                # Compute loss
                loss, info = model.compute_loss(batch_data, forward_result)
                # calculate averge iou
                
                train_losses.append(loss.item())
                bar.set_postfix(
                    **info
                )
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

                pass

            # Validation process
            model.eval()
            val_losses = []
            info = dict()
            gt_records = []
            pred_records = []
            num_images = 0
            for i_batch, batch_data in enumerate(dataloader_val):
                batch_data = torch_utils.batch_to_device(batch_data, device)

                with torch.no_grad():
                    forward_result = model(batch_data)
                    # Compute loss
                    loss, info_ = model.compute_loss(batch_data, forward_result)
                    preds = model.postprocess(forward_result, batch_data)

                    for pred, image_id in zip(preds, batch_data['image_id']):
                        pred.image_id = image_id
                        pass

                    info = add_stats(info, info_)
                    val_losses.append(loss.item())
                    # calculate averge iou
                    pass
                
                pred_records.extend(preds)
                gt_records.extend(extract_ground_truth(batch_data))
                pass
            
            info = divide_stats(info, len(dataloader_val))

            # Evaluate the model
            stat = evaluate.eval_detection_result(
                gt_records, pred_records, model.get_class_names())
            
            self.logger.info(
                f'Epoch {i_epoch}, lr: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses)}, validation loss: {np.mean(val_losses)}, '
                f'{format_stats(info)}'
            )

            if i_epoch >= warmup_epochs:
                scheduler.step()
                pass


            # Save checkpoint

            checkpoint_path = os.path.join(self.args.checkpoint_path, 'ckpt')
            model.save(checkpoint_path)
            pass
        pass


    pass