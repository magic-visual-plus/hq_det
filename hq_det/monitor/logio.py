import sys
import os

class LogRedirector:
    def __init__(self, log_file_path=None):
        """
        初始化日志重定向器，将标准输出和标准错误同时输出到控制台和日志文件
        
        Args:
            log_file_path: 日志文件路径，如果为None则不进行重定向
        """
        self.log_file = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.tee_stdout = None
        self.tee_stderr = None
        
        if log_file_path:
            self.redirect(log_file_path)
    
    def redirect(self, log_file_path):
        """
        重定向输出到日志文件
        
        Args:
            log_file_path: 日志文件路径
        """
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 打开日志文件
        self.log_file = open(log_file_path, 'w')
        
        # 创建Tee输出对象并重定向
        self.tee_stdout = self._TeeOutput(self.log_file, self.original_stdout)
        self.tee_stderr = self._TeeOutput(self.log_file, self.original_stderr)
        
        # 重定向标准输出和标准错误
        sys.stdout = self.tee_stdout
        sys.stderr = self.tee_stderr
        
    def restore(self):
        """恢复原始的标准输出和标准错误"""
        if self.log_file:
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            self.log_file.close()
            self.log_file = None
            self.tee_stdout = None
            self.tee_stderr = None
    
    class _TeeOutput:
        """内部类，用于同时将输出写入文件和原始输出流"""
        def __init__(self, file, original_output):
            self.file = file
            self.original_output = original_output
            
        def write(self, data):
            self.file.write(data)
            self.original_output.write(data)
            
        def flush(self):
            self.file.flush()
            self.original_output.flush()