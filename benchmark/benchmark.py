import sys
import os
from typing import List
import torch
import gc
from utils.calc_fps import official_fps_test_forward, official_fps_test_predict
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
# from hq_det.tools.train_codetr import MyTrainer as CODETR_Trainer
# from hq_det.tools.train_rfdetr import MyTrainer as RRFDETR_Trainer
from hq_det.tools.train_rtdetr import MyTrainer as RDETR_Trainer
# from hq_det.tools.train_rtmdet import MyTrainer as RTMDet_Trainer
from hq_det.tools.train_dino import MyTrainer as DINO_Trainer
# from hq_det.tools.train_yolo import MyTrainer as YOLO_Trainer



def benchmark(model_path, data_path, output_path, eval_class_names = []):
    pass


def fps_benchmark(model_paths: dict, data_path: str, eval_class_names = [], img_paths: List[str] = None, batch_size=1, total_tests=100, iscompile=False):
    output_path = "output/fps_benchmark"
    trainer_cls = {
        # "codetr": CODETR_Trainer,
        # "rfdetr": RRFDETR_Trainer,
        "rtdetr": RDETR_Trainer,
        # "rtmdet": RTMDet_Trainer,
        "dino": DINO_Trainer,
        # "yolo": YOLO_Trainer
    }
    trainer_configs = {}
    if "codetr" in model_paths:
        trainer_configs["codetr"] = get_codetr_config(data_path, model_paths["codetr"], output_path, eval_class_names, batch_size=batch_size)
    if "rfdetr" in model_paths:
        trainer_configs["rfdetr"] = get_rfdetr_config(data_path, model_paths["rfdetr"], output_path, eval_class_names, batch_size=batch_size)
    if "rtdetr" in model_paths:
        trainer_configs["rtdetr"] = get_rtdetr_config(data_path, model_paths["rtdetr"], output_path, eval_class_names, batch_size=batch_size)
    if "rtmdet" in model_paths:
        trainer_configs["rtmdet"] = get_rtmdet_config(data_path, model_paths["rtmdet"], output_path, eval_class_names, batch_size=batch_size)
    if "dino" in model_paths:
        trainer_configs["dino"] = get_dino_config(data_path, model_paths["dino"], output_path, eval_class_names, batch_size=batch_size)
    if "yolo" in model_paths:
        trainer_configs["yolo"] = get_yolo_config(data_path, model_paths["yolo"], output_path, eval_class_names, batch_size=batch_size)

    fps_dict = {"forward": {}, "predict": {}}
    for model_name in model_paths.keys():
        try:
            print(f"Benchmarking {model_name}...")
            
            torch.cuda.empty_cache()
            gc.collect()
            
            trainer = trainer_cls[model_name](trainer_configs[model_name])
            trainer.setup_training_environment()
            trainer.model.eval()  
            # print(trainer.model)
            # 
            if iscompile:
                if model_name == "dino":
                    trainer.model.model.data_preprocessor = torch._dynamo.disable(trainer.model.model.data_preprocessor)
                trainer.model.model = torch.compile(trainer.model.model, dynamic=False)
            
            batch_data = next(iter(trainer.dataloader_val))
            batch_data = batch_to_device(batch_data, trainer.device)

            # 测试predict FPS (包含后处理)
            print(f"Calculating {model_name} predict fps...")
            predict_fps = official_fps_test_predict(trainer.model, img_paths, precision="FP32", max_size=1024, batch_size=batch_size, total_tests=total_tests)
            fps_dict["predict"][model_name] = predict_fps
            
            # 测试forward FPS
            print(f"Calculating {model_name} forward fps...")
            forward_fps = official_fps_test_forward(trainer.model, batch_data, precision="FP32", batch_size=batch_size, total_tests=total_tests)
            fps_dict["forward"][model_name] = forward_fps
            
        except Exception as e:
            print(f"Error benchmarking {model_name}: {str(e)}")
            fps_dict["forward"][model_name] = None
            fps_dict["predict"][model_name] = None
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
        # "codetr": "/root/autodl-tmp/model/codetr/co_dino_5scale_r50_1x_coco-7481f903.pth",
        # "rfdetr": "/root/autodl-tmp/model/rfdetr/rf-detr-base.pth",
        "rtdetr": "/root/autodl-tmp/model/rtdetrv2/rtdetrv2_r50vd_m_7x_coco_ema.pth",
        # "rtmdet": "/root/autodl-tmp/model/rtmdet/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth",
        # "dino": "/root/autodl-tmp/model/dino/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth",
        # "yolo": "/root/autodl-tmp/model/yolo/yolov12l.pt"
    }
    data_path = sys.argv[1]
    img_path = os.path.join(data_path, "valid")
    img_paths = [os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith('.jpg') or f.endswith('.png')]
    print(img_paths[0:1])
    fps_dict = fps_benchmark(model_paths, data_path, img_paths=img_paths[0:16], batch_size=16, total_tests=100, iscompile=False)
    print(fps_dict)


    






