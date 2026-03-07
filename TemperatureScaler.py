import torch
from torch import nn
import torch.optim as optim

class TemperatureScaler(nn.Module):
    """
    Temperature Scaling for probability calibration
    Reference: Guo et al. (ICML 2017)
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature.clamp(min=1e-6)

    def fit(self, logits, labels, max_iter=50, lr=0.01):
        """
        logits: torch.Tensor [N, C]
        labels: torch.Tensor [N]
        """
        self.to(logits.device)
        nll = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self
