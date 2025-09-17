import torch
import torch.nn as nn
import torch.nn.functional as F

def get_ratio(teacher_logits, logits, mu=0.5):
    # 处理无穷值
    teacher_logits = torch.masked_fill(teacher_logits, torch.isinf(teacher_logits), 0).to(torch.float32)
    logits = torch.masked_fill(logits, torch.isinf(logits), 0).to(torch.float32)
    
    # 计算概率分布
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32).detach()
    
    # 排序概率
    re_teacher_probs, idx = teacher_probs.sort(dim=-1, descending=True)
    re_student_probs = student_probs.gather(dim=-1, index=idx)
    
    # 计算误差
    errors = torch.abs(re_teacher_probs - re_student_probs)
    
    # 计算累积概率并创建掩码
    cum_sum = torch.cumsum(re_teacher_probs, dim=-1)
    mask = cum_sum > mu
    mask[:,:,0] = False
    
    # 计算高低概率区域的误差
    s1 = torch.masked_fill(errors, mask, 0.0).sum(dim=-1)
    s2 = torch.masked_fill(errors, ~mask, 0.0).sum(dim=-1)
    
    # 避免除零错误
    total = s1 + s2
    total[total == 0] = 1e-10
    
    return s1/total, s2/total

def get_kl_loss(teacher_logits, logits, inf_mask, mask, ratio=None):
    # 计算正向KL散度
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    teacher_prod_probs = torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    teacher_x = torch.sum(teacher_prod_probs, dim=-1).view(-1)
    
    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    
    if ratio is None:
        fkl_loss = torch.sum((teacher_x - x) * mask.view(-1), dim=0) / (torch.sum(mask.view(-1), dim=0) + 1e-10)
    else:
        fkl_loss = torch.sum((teacher_x - x) * ratio.view(-1) * mask.view(-1), dim=0) / (torch.sum(mask.view(-1), dim=0) + 1e-10)
    
    # 计算反向KL散度
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    student_prod_probs = torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    student_x = torch.sum(student_prod_probs, dim=-1).view(-1)
    
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    prod_probs_rev = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    x_rev = torch.sum(prod_probs_rev, dim=-1).view(-1)
    
    if ratio is None:
        rkl_loss = torch.sum((student_x - x_rev) * mask.view(-1), dim=0) / (torch.sum(mask.view(-1), dim=0) + 1e-10)
    else:
        rkl_loss = torch.sum((student_x - x_rev) * ratio.view(-1) * mask.view(-1), dim=0) / (torch.sum(mask.view(-1), dim=0) + 1e-10)
    
    return fkl_loss, rkl_loss