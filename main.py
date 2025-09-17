import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from transformers import GPT2Tokenizer

# 添加路径
sys.path.append('./models')
sys.path.append('./loss')
sys.path.append('./rl_agent')
sys.path.append('./utils')

from config import config
from models import StudentModel, TeacherModel, PolicyNetwork
from loss import get_ratio, get_kl_loss
from rl_agent import ReinforceAgent
from utils import get_dolly_data_loader, evaluate_model

def main():
    # 初始化标记器
    tokenizer = GPT2Tokenizer.from_pretrained(config.teacher_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 初始化模型
    teacher_model = TeacherModel(config).to(config.device)
    student_model = StudentModel(config).to(config.device)
    policy_net = PolicyNetwork(input_size=10).to(config.device)
    reinforce_agent = ReinforceAgent(policy_net, config)
    
    # 数据加载器
    train_loader = get_dolly_data_loader(config, tokenizer, "train")
    val_loader = get_dolly_data_loader(config, tokenizer, "validation")
    
    # 优化器
    optimizer = optim.Adam(student_model.parameters(), lr=config.learning_rate)
    
    # 训练循环
    best_rouge = 0
    for epoch in range(config.num_epochs):
        student_model.train()
        total_loss = 0
        total_rl_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["labels"].to(config.device)
            
            # 前向传播
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                teacher_logits = teacher_outputs.logits
            
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            student_logits = student_outputs.logits
            
            # 计算KL损失
            inf_mask = torch.isinf(student_logits)
            mask = (labels != -100).int()
            
            h_ratio, l_ratio = get_ratio(teacher_logits, student_logits)
            fkl_loss, rkl_loss = get_kl_loss(teacher_logits, student_logits, inf_mask, mask, h_ratio)
            
            # 获取状态
            state = np.array([
                epoch/config.num_epochs,
                batch_idx/len(train_loader),
                fkl_loss.item(),
                rkl_loss.item(),
                h_ratio.mean().item(),
                l_ratio.mean().item(),
                student_outputs.loss.item(),
                0, 0, 0  # 预留位置用于其他指标
            ])
            
            # 策略网络选择α
            alpha = reinforce_agent.select_action(state)
            
            # 计算混合损失
            mixed_loss = alpha * fkl_loss + (1 - alpha) * rkl_loss
            
            # 反向传播更新学生模型
            optimizer.zero_grad()
            mixed_loss.backward()
            optimizer.step()
            
            total_loss += mixed_loss.item()
            
            # 定期评估并更新策略网络
            if batch_idx % 100 == 0 and batch_idx > 0:
                # 评估模型
                rouge_score = evaluate_model(
                    student_model, teacher_model, tokenizer, val_loader, config.device, config
                )
                
                # 计算奖励（ROUGE-L分数）
                reward = rouge_score - 0.5  # 基准奖励
                reinforce_agent.rewards.append(reward)
                
                # 更新策略网络
                if len(reinforce_agent.rewards) >= 5:  # 每5步更新一次
                    rl_loss = reinforce_agent.update_policy()
                    total_rl_loss += rl_loss
                
                print(f"Epoch {epoch+1}, Batch {batch_idx}, ROUGE-L: {rouge_score:.4f}, Reward: {reward:.4f}")
        
        # 打印 epoch 统计信息
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {avg_loss:.4f}, RL Loss: {total_rl_loss:.4f}")
        
        # 评估模型
        rouge_score = evaluate_model(
            student_model, teacher_model, tokenizer, val_loader, config.device, config
        )
        print(f"Epoch {epoch+1}, ROUGE-L: {rouge_score:.4f}")
        
        # 保存最佳模型
        if rouge_score > best_rouge:
            best_rouge = rouge_score
            torch.save(student_model.state_dict(), os.path.join(config.save_dir, "best_student_model.pth"))
            torch.save(policy_net.state_dict(), os.path.join(config.save_dir, "best_policy_net.pth"))
        
        # 保存检查点
        torch.save(student_model.state_dict(), os.path.join(config.save_dir, f"student_model_epoch_{epoch+1}.pth"))
        torch.save(policy_net.state_dict(), os.path.join(config.save_dir, f"policy_net_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)
    main()