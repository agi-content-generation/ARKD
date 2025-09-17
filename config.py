import torch

class Config:
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 训练参数
    batch_size = 4
    num_epochs = 3
    learning_rate = 5e-5
    rl_learning_rate = 1e-4
    
    # 模型参数
    teacher_model_name = "gpt2-medium"
    student_model_name = "gpt2"
    
    # 强化学习参数
    gamma = 0.99  # 折扣因子
    entropy_coef = 0.01  # 熵正则化系数
    
    # 文本生成参数
    max_length = 128
    num_beams = 4
    
    # 路径配置
    save_dir = "checkpoints/"
    
config = Config()