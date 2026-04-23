import os
from config_loader import ConfigLoader

def convert_all_scripts():
    # 获取脚本目录和配置目录
    script_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts')
    config_dir = os.path.dirname(__file__)
    
    # 创建配置加载器实例
    loader = ConfigLoader(config_dir)
    
    # 遍历所有训练脚本
    for filename in os.listdir(script_dir):
        if filename.endswith('_train.sh'):
            # 构建输入和输出路径
            script_path = os.path.join(script_dir, filename)
            config_name = filename.replace('_train.sh', '')
            output_path = os.path.join(config_dir, f"{config_name}.yaml")
            
            # 转换脚本为YAML
            print(f"Converting {filename} to {config_name}.yaml")
            loader.shell_to_yaml(script_path, output_path)

if __name__ == '__main__':
    convert_all_scripts() 