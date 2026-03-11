import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, labels):
        bce_loss_func = F.binary_cross_entropy_with_logits(preds, labels, reduction='none')
        pt = torch.exp(-bce_loss_func)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss_func

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss