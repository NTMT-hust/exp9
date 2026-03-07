from torch import nn
import torch

class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=2.0,
                smoothing=0.1, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = torch.log_softmax(logits, dim=1)

        # Label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Cross entropy with smoothed labels
        ce_loss = -torch.sum(true_dist * log_probs, dim=1)

        # Focal term
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Class weighting
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
