import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.header import Header
import os
import time

class EmailSender:
    def __init__(self, sender_email, sender_password, smtp_server="smtp.163.com", smtp_port=25):
        """
        初始化邮件发送器
        :param sender_email: 发件人邮箱
        :param sender_password: 邮箱授权码
        :param smtp_server: SMTP服务器地址
        :param smtp_port: SMTP服务器端口
        """
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    def _create_base_message(self, receiver_email, subject):
        """
        创建基础邮件对象
        :param receiver_email: 收件人邮箱，可以是单个邮箱字符串或邮箱列表
        :param subject: 邮件主题
        :return: 邮件对象和收件人列表
        """
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        
        # 处理收件人，可能是字符串或列表
        if isinstance(receiver_email, list):
            msg['To'] = ', '.join(receiver_email)
            recipients = receiver_email
        else:
            msg['To'] = receiver_email
            recipients = [receiver_email]
            
        msg['Subject'] = Header(subject, 'utf-8')
        return msg, recipients

    def _add_content(self, msg, content, content_type='html'):
        """
        添加邮件内容
        :param msg: 邮件对象
        :param content: 邮件内容
        :param content_type: 内容类型，'plain' 或 'html'
        """
        msg.attach(MIMEText(content, content_type, 'utf-8'))
        return msg

    def _add_attachment(self, msg, attachment_path):
        """
        添加附件
        :param msg: 邮件对象
        :param attachment_path: 附件路径
        :return: 添加附件后的邮件对象
        """
        if attachment_path and os.path.exists(attachment_path):
            try:
                with open(attachment_path, 'rb') as file:
                    attachment = MIMEApplication(file.read())
                    filename = os.path.basename(attachment_path)
                    attachment.add_header('Content-Disposition', 'attachment', filename=filename)
                    msg.attach(attachment)
                    print(f"已添加文件：{filename}作为附件")
                return True
            except Exception as e:
                print(f"添加附件失败：{str(e)}")
                return False
        return False

    def _send_email(self, msg, recipients):
        """
        发送邮件
        :param msg: 邮件对象
        :param recipients: 收件人列表
        """
        try:
            # 连接SMTP服务器
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            
            # 发送邮件
            server.sendmail(self.sender_email, recipients, msg.as_string())
            print("邮件发送成功！")
            return True
            
        except Exception as e:
            print(f"邮件发送失败：{str(e)}")
            return False
        finally:
            server.quit()

    def _format_additional_info(self, additional_info):
        """
        格式化额外信息
        :param additional_info: 额外信息
        :return: 格式化后的信息
        """
        if isinstance(additional_info, str):
            # 替换换行符为HTML的<br>标签
            formatted_info = additional_info.replace('\n', '<br>')
            # 替换制表符为空格
            formatted_info = formatted_info.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')
            # 替换回车符
            formatted_info = formatted_info.replace('\r', '')
            return formatted_info
        return str(additional_info)

    def send_experiment_notification(self, receiver_email, experiment_name, attachments=None, training_time=None, additional_info=None):
        """
        发送实验完成通知邮件
        :param receiver_email: 收件人邮箱，可以是单个邮箱字符串或邮箱列表
        :param experiment_name: 实验名称
        :param attachments: 附件文件路径，可以是单个文件路径或文件路径列表
        :param training_time: 训练时长（可选）
        :param additional_info: 额外信息（可选）
        """
        # 创建邮件基础对象
        msg, recipients = self._create_base_message(
            receiver_email, 
            f'实验完成通知 - {experiment_name}'
        )

        # 构建邮件正文
        content = f"""
        <html>
        <body>
            <h2>实验完成通知</h2>
            <p>实验名称：{experiment_name}</p>
            <p>完成时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}</p>
        """
        
        if training_time:
            content += f"<p>训练时长：{training_time}</p>"
        
        if additional_info:
            formatted_info = self._format_additional_info(additional_info)
            content += f"<p>{formatted_info}</p>"
        
        content += """
        </body>
        </html>
        """

        # 添加HTML内容
        self._add_content(msg, content, 'html')
        
        # 添加附件
        if attachments:
            if isinstance(attachments, list):
                for attachment_path in attachments:
                    self._add_attachment(msg, attachment_path)
            else:
                self._add_attachment(msg, attachments)

        # 发送邮件
        return self._send_email(msg, recipients)

    def send_custom_email(self, receiver_email, subject, content, content_type='plain', attachments=None):
        """
        发送自定义邮件
        :param receiver_email: 收件人邮箱，可以是单个邮箱字符串或邮箱列表
        :param subject: 邮件主题
        :param content: 邮件内容
        :param content_type: 内容类型，'plain' 或 'html'
        :param attachments: 附件路径列表
        """
        # 创建邮件基础对象
        msg, recipients = self._create_base_message(receiver_email, subject)
        
        # 添加内容
        self._add_content(msg, content, content_type)
        
        # 添加附件
        if attachments:
            for attachment_path in attachments:
                self._add_attachment(msg, attachment_path)
        
        # 发送邮件
        return self._send_email(msg, recipients)

# 使用示例
if __name__ == "__main__":
    import socket
    import argparse
    import ast

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='发送邮件通知')
    parser.add_argument('--attachments', type=str, default="None", help='附件文件路径，可以是单个文件路径或文件路径列表如[path1, path2]')
    parser.add_argument('--receiver', type=str, default="example@example.com", help='接收者邮箱，可以是单个邮箱或邮箱列表如[email1, email2]')
    parser.add_argument('--sender', type=str, default="sender@example.com", help='发送者邮箱')
    parser.add_argument('--password', type=str, default="your_password_here", help='发送者邮箱授权码')
    parser.add_argument('--subject', type=str, default="任务完成通知", help='邮件主题')
    parser.add_argument('--additional_info', type=str, default="任务已完成", help='额外信息')
    
    args = parser.parse_args()
    
    # 处理接收者邮箱，可能是字符串或列表字符串
    receiver = args.receiver
    if receiver.startswith('[') and receiver.endswith(']'):
        try:
            # 尝试将字符串解析为列表
            receiver = ast.literal_eval(receiver)
        except (SyntaxError, ValueError):
            # 如果解析失败，尝试简单的逗号分隔解析
            if ',' in receiver:
                # 去掉方括号，按逗号分隔，并去除每个元素的空白
                receiver = [email.strip() for email in receiver[1:-1].split(',')]
    
    # 处理附件，可能是字符串或列表字符串
    attachments = args.attachments
    if attachments != "None":
        if attachments.startswith('[') and attachments.endswith(']'):
            try:
                # 尝试将字符串解析为列表
                attachments = ast.literal_eval(attachments)
            except (SyntaxError, ValueError):
                # 如果解析失败，尝试简单的逗号分隔解析
                if ',' in attachments:
                    # 去掉方括号，按逗号分隔，并去除每个元素的空白
                    attachments = [path.strip() for path in attachments[1:-1].split(',')]
        else:
            attachments = [attachments]
    else:
        attachments = None
    
    # 初始化邮件发送器
    sender = EmailSender(
        sender_email=args.sender,
        sender_password=args.password
    )
    
    # 获取服务器IP地址
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    
    # 发送通知
    sender.send_experiment_notification(
        receiver_email=receiver,
        experiment_name=args.subject,
        attachments=attachments,
        additional_info=f"{args.additional_info}\n服务器IP: {ip_address}"
    )
