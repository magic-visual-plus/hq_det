import torch  
import time  
import numpy as np  
from tqdm import tqdm
from typing import List
import cv2

def official_fps_test_forward(model, batch_data, precision="FP16", batch_size=1, total_tests=1000):  
    """  
    official FPS test  
    """  
    model.eval()
    model.to("cuda:0")
    
    print("="*50)  
    print("official FPS test")  
    print("="*50)  
    print(f"GPU: {torch.cuda.get_device_name()}")  
    print(f"CUDA: {torch.version.cuda}")  
    print(f"PyTorch: {torch.__version__}")  
    
    print("warming up...")  
    for _ in range(5):  
        with torch.inference_mode():  
            _ = model(batch_data)  
      
    print("start FPS test...")  
    times = []  
    
    for i in tqdm(range(total_tests), desc="FPS test progress"):  
        torch.cuda.synchronize()  
        start = time.perf_counter()  
        
        with torch.inference_mode():
            output = model(batch_data)  
        
        torch.cuda.synchronize()  
        end = time.perf_counter()  
        
        times.append(end - start)  
    
    drop_count = len(times) // 10
    times = sorted(times)[drop_count:-drop_count]  # drop the first 10% and the last 10%  
    avg_time = np.mean(times)  
    std_time = np.std(times)  
    
    # 7. 输出结果  
    print("\n" + "="*40)  
    print("test result")  
    print("="*40)  
    print(f"average inference time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")  
    print(f"FPS: {1.0/avg_time:.1f}")  
    print(f"input size: 1024×1024")  
    print(f"precision: {precision}")  
    print(f"batch size: {batch_size}")  
    
    return 1.0 / avg_time  

def official_fps_test_predict(model, img_paths: List[str], precision="FP16", max_size=1024, batch_size=1, total_tests=1000):
    """  
    official FPS test for predict (包含后处理)
    """  
    print("="*50)  
    print("official FPS test for predict")  
    print("="*50)  
    print(f"GPU: {torch.cuda.get_device_name()}")  
    print(f"CUDA: {torch.version.cuda}")  
    print(f"PyTorch: {torch.__version__}")  
    
    model.eval()
    model.to("cuda:0")
    
    # 过滤图片文件
    filenames = [f for f in img_paths if f.endswith('.jpg') or f.endswith('.png')]
    
    if len(filenames) == 0:
        print("No valid image files found!")
        return 0.0
    
    print("warming up...")  
    # 预热阶段
    for _ in range(5):  
        imgs = [cv2.imread(filenames[i]) for i in range(batch_size)]
        with torch.inference_mode():
            _ = model.predict(imgs, bgr=True, confidence=0.3, max_size=max_size)
    
    print("start FPS test...")  
    times = []
    
    # 如果图片数量不足1000张，则循环使用
    for i in tqdm(range(total_tests), desc="FPS test progress"):
        # 循环使用图片
        img_indices = [(i * batch_size + j) % len(filenames) for j in range(batch_size)]
        batch_filenames = [filenames[idx] for idx in img_indices]
        
        imgs = [cv2.imread(filename) for filename in batch_filenames]
            
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.inference_mode():
            results = model.predict(imgs, bgr=True, confidence=0.3, max_size=max_size)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append(end - start)
    
    if len(times) < 2:
        print("Warning: Not enough valid test samples!")
        return 0.0
    
    # 去掉前10%和后10%的数据
    times = sorted(times)
    drop_count = len(times) // 10
    times = times[drop_count:-drop_count]
    
    avg_time = np.mean(times)  
    std_time = np.std(times)  
    
    # 输出结果
    print("\n" + "="*40)  
    print("test result")  
    print("="*40)  
    print(f"average inference time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")  
    print(f"FPS: {1.0/avg_time:.1f}")  
    print(f"precision: {precision}")  
    print(f"total test images: {len(filenames)}")  
    print(f"total test runs: {len(times)}")  
    
    return 1.0 / avg_time