import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class TeacherModel(nn.Module):
    def __init__(self, config):
        super(TeacherModel, self).__init__()
        self.config = config
        
        # 加载预训练模型
        self.model = GPT2LMHeadModel.from_pretrained(config.teacher_model_name)
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 设置为评估模式
        self.model.eval()
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        with torch.no_grad():  # 确保不计算梯度
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        return outputs
    
    def generate(self, input_ids, attention_mask=None, **kwargs):
        with torch.no_grad():  # 确保不计算梯度
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.config.max_length,
                num_beams=self.config.num_beams,
                **kwargs
            )