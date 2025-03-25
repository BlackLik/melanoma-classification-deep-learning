import torch
from torch import nn
from torchvision import models


class ResNet18Model(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)

    def apply_sparsity(self, sparsity_rate=0.5):
        """
        Применяет динамическую разреженность.
        sparsity_rate: процент обнулённых весов
        """
        with torch.no_grad():
            weight = self.model.fc.weight.data
            abs_weight = torch.abs(weight)
            threshold = torch.quantile(abs_weight, sparsity_rate)
            weight[abs_weight < threshold] = 0  # Обнуляем веса ниже порога
