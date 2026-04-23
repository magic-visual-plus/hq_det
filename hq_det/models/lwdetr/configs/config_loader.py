import os
import yaml
import argparse
from typing import Dict, Any, Optional

class ConfigLoader:
    def __init__(self, config_dir: str):
        """
        初始化配置加载器
        Args:
            config_dir: YAML配置文件所在目录
        """
        self.config_dir = config_dir

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        加载指定的配置文件，先加载默认配置，再加载指定配置进行覆盖
        Args:
            config_name: 配置文件名（不含.yaml后缀）
        Returns:
            配置字典
        """
        # 首先加载默认配置
        default_config_path = os.path.join(self.config_dir, "lw_detr_default_args.yaml")
        if not os.path.exists(default_config_path):
            raise FileNotFoundError(f"Default config file not found: {default_config_path}")
        
        with open(default_config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 然后加载指定配置进行覆盖
        config_path = os.path.join(self.config_dir, f"{config_name}.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            override_config = yaml.safe_load(f)
            config.update(override_config)
            
        return config

    def get_args(self, config_name: str, additional_args: Optional[Dict[str, Any]] = None) -> argparse.Namespace:
        """
        将配置转换为argparse.Namespace对象
        Args:
            config_name: 配置文件名（不含.yaml后缀）
            additional_args: 额外的命令行参数
        Returns:
            argparse.Namespace对象
        """
        config = self.load_config(config_name)
        
        # 合并额外参数
        if additional_args:
            config.update(additional_args)
        
        # 确保数字类型参数不会被加载成字符串
        for key, value in config.items():
            if isinstance(value, str):
                try:
                    if value.isdigit():
                        config[key] = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        config[key] = float(value)
                    # 处理科学计数法形式的字符串
                    elif 'e' in value.lower():
                        config[key] = float(value)
                except ValueError:
                    pass
        
        # 转换为Namespace对象
        args = argparse.Namespace(**config)
        return args

    @staticmethod
    def shell_to_yaml(shell_script_path: str, output_yaml_path: str):
        """
        将shell脚本中的参数转换为YAML文件
        Args:
            shell_script_path: shell脚本路径
            output_yaml_path: 输出YAML文件路径
        """
        with open(shell_script_path, 'r') as f:
            content = f.read()

        args = {}
        for line in content.split('\n'):
            if line.startswith('    --'):
                parts = line.strip().split(' ', 1)
                key = parts[0].replace('--', '')
                value = None
                if len(parts) == 2:
                    value = parts[1].strip()
                    # 去除末尾的 \\ 和多余空格
                    if value.endswith('\\'):
                        value = value[:-2].strip()
                    # 处理列表
                    if value:
                        if ' ' in value:
                            value = [int(x) if x.isdigit() else float(x) if x.replace('.', '', 1).isdigit() else x for x in value.split()]
                        else:
                            # 类型转换
                            if value.lower() == 'true':
                                value = True
                            elif value.lower() == 'false':
                                value = False
                            elif value.isdigit():
                                value = int(value)
                            elif value.replace('.', '', 1).isdigit():
                                value = float(value)
                    else:
                        value = True  # 仅有参数名无值，视为True
                else:
                    value = True  # 仅有参数名无值，视为True
                args[key] = value

        with open(output_yaml_path, 'w') as f:
            yaml.dump(args, f, default_flow_style=False, sort_keys=False, allow_unicode=True) 


if __name__ == "__main__":
    config_loader = ConfigLoader(config_dir=os.path.dirname(__file__))
    args = config_loader.get_args("lwdetr_large_coco")
    print(args)
