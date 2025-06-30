import torch  
import time  
import numpy as np  
from tqdm import tqdm


def official_fps_test_template(model, batch_data, precision="FP16", batch_size=1):  
    """  
    official FPS test  
    """  
    print("="*50)  
    print("official FPS test")  
    print("="*50)  
    print(f"GPU: {torch.cuda.get_device_name()}")  
    print(f"CUDA: {torch.version.cuda}")  
    print(f"PyTorch: {torch.__version__}")  
    
    print("warming up...")  
    for _ in range(100):  
        with torch.no_grad():  
            _ = model(batch_data)  
      
    print("start FPS test...")  
    times = []  
    
    for i in tqdm(range(1000), desc="FPS test progress"):  
        torch.cuda.synchronize()  
        start = time.perf_counter()  
        
        with torch.no_grad():  
            output = model(batch_data)  
        
        torch.cuda.synchronize()  
        end = time.perf_counter()  
        
        times.append(end - start)  
    
    times = sorted(times)[100:900]  # drop the first 10% and the last 10%  
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
