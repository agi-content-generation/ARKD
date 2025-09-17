import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

class StudentModel(nn.Module):
    def __init__(self, config):
        super(StudentModel, self).__init__()
        self.config = config
        
        # 加载预训练模型
        self.model = GPT2LMHeadModel.from_pretrained(config.student_model_name)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def generate(self, input_ids, attention_mask=None, **kwargs):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.config.max_length,
            num_beams=self.config.num_beams,
            **kwargs
        )