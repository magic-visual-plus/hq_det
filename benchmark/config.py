from hq_det.common import HQTrainerArguments


def get_codetr_config(data_path, model_path, output_path, eval_class_names = [], batch_size = None):
    return HQTrainerArguments(
        data_path=data_path,
        num_epoches=20,
        warmup_epochs=2,
        num_data_workers=8,
        lr0=1e-3,
        lr_min=1e-5,
        batch_size=batch_size if batch_size is not None else 4,
        device='cuda:0',
        checkpoint_path=output_path,
        output_path=output_path,
        checkpoint_interval=1,
        image_size=1024,
        model_argument={
            "model": model_path,
        },
        eval_class_names=eval_class_names,
    )

def get_rfdetr_config(data_path, model_path, output_path, eval_class_names = [], model_type = "base", batch_size = None):
    return  HQTrainerArguments(
            data_path=data_path,  
            num_epoches=125,  
            warmup_epochs=10, 
            num_data_workers=8, 
            lr0=1e-4, 
            lr_min=1e-6, 
            lr_backbone_mult=0.1, 
            batch_size=batch_size if batch_size is not None else 4, 
            device='cuda:0', 
            checkpoint_path=output_path, 
            output_path=output_path, 
            checkpoint_interval=1, 
            image_size=1024, 
            gradient_update_interval=1, 
            max_grad_norm=0.1, 
            model_argument={
                "model": model_path, 
                "model_type": model_type, 
                "lr_encoder": 1.5e-4, 
                "lr_component_decay": 0.7, 
            },
            eval_class_names=eval_class_names,  
            early_stopping=False, 
            early_stopping_patience=10, 
        )

def get_rtdetr_config(data_path, model_path, output_path, eval_class_names = [], batch_size = None):
    return HQTrainerArguments(
            data_path=data_path,
            num_epoches=180,
            warmup_epochs=2,
            num_data_workers=8,
            lr0=1e-4,
            lr_min=1e-6,
            lr_backbone_mult=1,
            batch_size=batch_size if batch_size is not None else 6,
            device='cuda:0',
            output_path=output_path,
            checkpoint_path=output_path,
            checkpoint_interval=-1,
            image_size=1024,
            model_argument={
                "model": model_path,
            },
            eval_class_names=eval_class_names,
            gradient_update_interval=1,
        )

def get_rtmdet_config(data_path, model_path, output_path, eval_class_names = [], batch_size = None):
    return HQTrainerArguments(
            data_path=data_path,
            num_epoches=180,
            warmup_epochs=2,
            num_data_workers=8,
            lr0=1e-3,
            lr_min=1e-5,
            lr_backbone_mult=0.1,
            batch_size=batch_size if batch_size is not None else 4,
            device='cuda:0',
            checkpoint_path=output_path,
            output_path=output_path,
            checkpoint_interval=1,
            image_size=1024,
            model_argument={
                "model": model_path,
            },
            eval_class_names=eval_class_names,
            gradient_update_interval=1,
        )

def get_dino_config(data_path, model_path, output_path, eval_class_names = [], batch_size = None):
    return HQTrainerArguments(
        data_path=data_path,
        num_epoches=100,
        warmup_epochs=2,
        num_data_workers=8,
        lr0=4e-4,
        lr_min=1e-6,
        lr_backbone_mult=1.0,
        batch_size=batch_size if batch_size is not None else 4,
        device='cuda:0',
        checkpoint_path=output_path,
        output_path=output_path,
        checkpoint_interval=1,
        gradient_update_interval=4,
        image_size=1024,
        model_argument={
            "model": model_path,
        },
        eval_class_names=eval_class_names,
    )

def get_yolo_config(data_path, model_path, output_path, eval_class_names = [], batch_size = None):
    return HQTrainerArguments(
        data_path=data_path,
        num_epoches=125,
        warmup_epochs=10,
        num_data_workers=8,
        lr0=1e-4,
        lr_min=1e-6,
        lr_backbone_mult=1,
        batch_size=batch_size if batch_size is not None else 4,
        device='cuda:0',
        checkpoint_path=output_path,
        output_path=output_path,
        checkpoint_interval=1,
        image_size=1024,
        model_argument={
            "model_path": model_path,
        },
        eval_class_names=eval_class_names,
        gradient_update_interval=1,
    )
