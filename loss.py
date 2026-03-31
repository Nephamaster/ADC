import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossCSC(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: [N, C], targets: [N]
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        # one-hot
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        targets_one_hot = torch.where(targets.unsqueeze(1) == self.ignore_index, 0.0, targets_one_hot)
        # focal weight
        pt = torch.sum(probs * targets_one_hot, dim=-1)
        focal_weight = (1 - pt) ** self.gamma
        # alpha balancing
        alpha_t = torch.where(targets == 1, 1 - self.alpha, self.alpha)  # index 1 = correct char (NamBert style)
        # 但更通用做法：对所有非 ignore 的 token 使用 alpha
        # alpha_t = torch.where(targets == self.ignore_index, 0.0, self.alpha)
        loss = -alpha_t * focal_weight * torch.sum(targets_one_hot * log_probs, dim=-1)
        return loss.mean()