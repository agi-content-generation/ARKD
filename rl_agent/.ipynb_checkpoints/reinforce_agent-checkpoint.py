import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ReinforceAgent:
    def __init__(self, policy_net, config):
        self.policy_net = policy_net
        self.optimizer = optim.Adam(policy_net.parameters(), lr=config.rl_learning_rate)
        self.gamma = config.gamma
        self.entropy_coef = config.entropy_coef
        self.saved_log_probs = []
        self.rewards = []
        
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        alpha = self.policy_net(state)
        dist = torch.distributions.Bernoulli(alpha)
        action = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action))
        return action.item()
    
    def update_policy(self):
        if not self.rewards:
            return 0
            
        R = 0
        policy_loss = []
        returns = []
        
        # 计算折扣回报
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        # 添加熵正则化
        entropy = torch.stack([-log_prob * torch.exp(log_prob) for log_prob in self.saved_log_probs]).sum()
        
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() - self.entropy_coef * entropy
        loss.backward()
        self.optimizer.step()
        
        # 清空缓存
        del self.rewards[:]
        del self.saved_log_probs[:]
        
        return loss.item()