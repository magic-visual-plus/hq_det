import sys
import os
import torch
import gc
from utils.calc_fps import official_fps_test_template
from hq_det.torch_utils import batch_to_device

from config import (
    get_codetr_config,  # codetr
    get_rfdetr_config,  # rfdetr
    get_rtdetr_config,  # rtdetr
    get_rtmdet_config,  # rtmdet
    get_dino_config,  # dino
    get_yolo_config  # yolo
)

# from hq_det.trainer import HQTrainer
from hq_det.tools.train_codetr import MyTrainer as CODETR_Trainer
from hq_det.tools.train_rfdetr import MyTrainer as RRFDETR_Trainer
from hq_det.tools.train_rtdetr import MyTrainer as RDETR_Trainer
from hq_det.tools.train_rtmdet import MyTrainer as RTMDet_Trainer
from hq_det.tools.train_dino import MyTrainer as DINO_Trainer
from hq_det.tools.train_yolo import MyTrainer as YOLO_Trainer



def benchmark(model_path, data_path, output_path, eval_class_names = []):
    pass


def fps_benchmark(model_paths: dict, data_path: str, eval_class_names = []):
    output_path = "output/fps_benchmark"
    batch_size = 1
    trainer_cls = {
        "codetr": CODETR_Trainer,
        "rfdetr": RRFDETR_Trainer,
        "rtdetr": RDETR_Trainer,
        "rtmdet": RTMDet_Trainer,
        "dino": DINO_Trainer,
        "yolo": YOLO_Trainer
    }
    trainer_configs = {
        "codetr": get_codetr_config(data_path, model_paths["codetr"], output_path, eval_class_names, batch_size=batch_size),
        "rfdetr": get_rfdetr_config(data_path, model_paths["rfdetr"], output_path, eval_class_names, batch_size=batch_size),
        "rtdetr": get_rtdetr_config(data_path, model_paths["rtdetr"], output_path, eval_class_names, batch_size=batch_size),
        "rtmdet": get_rtmdet_config(data_path, model_paths["rtmdet"], output_path, eval_class_names, batch_size=batch_size),
        "dino": get_dino_config(data_path, model_paths["dino"], output_path, eval_class_names, batch_size=batch_size),
        "yolo": get_yolo_config(data_path, model_paths["yolo"], output_path, eval_class_names, batch_size=batch_size)
    }

    fps_dict = {}
    for model_name in model_paths.keys():
        try:
            print(f"Benchmarking {model_name}...")
            
            torch.cuda.empty_cache()
            gc.collect()
            
            trainer = trainer_cls[model_name](trainer_configs[model_name])
            trainer.model.eval()  
            
            batch_data = next(iter(trainer.dataloader_val))
            batch_data = batch_to_device(batch_data, trainer.device)
            
            # 预热GPU
            print(f"Warming up {model_name}...")
            with torch.no_grad():
                for _ in range(10):
                    _ = trainer.model(batch_data)
            
            print(f"calculate {model_name} fps...")
            fps = official_fps_test_template(trainer.model, batch_data, precision="FP32", batch_size=batch_size)
            fps_dict[model_name] = fps
            
        except Exception as e:
            print(f"Error benchmarking {model_name}: {str(e)}")
            fps_dict[model_name] = None
        finally:
            if 'trainer' in locals():
                del trainer
            if 'batch_data' in locals():
                del batch_data
            
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    return fps_dict

if __name__ == "__main__":
    os.environ["HQ_DEBUG"] = "0"
    model_paths = {
        "codetr": "/root/autodl-tmp/model/codetr/co_dino_5scale_r50_1x_coco-7481f903.pth",
        "rfdetr": "/root/autodl-tmp/model/rfdetr/rf-detr-base.pth",
        "rtdetr": "/root/autodl-tmp/model/rtdetrv2/rtdetrv2_r50vd_m_7x_coco_ema.pth",
        "rtmdet": "/root/autodl-tmp/model/rtmdet/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth",
        "dino": "/root/autodl-tmp/model/dino/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth",
        "yolo": "/root/autodl-tmp/model/yolo/yolov12l.pt"
    }
    data_path = sys.argv[1]
    fps_dict = fps_benchmark(model_paths, data_path)
    print(fps_dict)


    






