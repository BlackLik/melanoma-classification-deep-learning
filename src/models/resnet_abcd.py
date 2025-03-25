import torch
from torch import nn
from torchvision import models

from .cosine_classifier import CosineClassifier


class ResNetCosineModel(nn.Module):
    def __init__(self, num_abcd_features=4, num_classes=2):
        super().__init__()
        # Используем resnet18 без последнего fc-слоя
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.cnn_fc = nn.Linear(512, 512)
        # Обработка ABCD-показателей через отдельный слой (если требуется)
        self.abcd_fc = nn.Linear(num_abcd_features, 512)
        # Вместо стандартного выходного слоя используем CosineClassifier,
        # который принимает объединённые признаки размерности 512
        self.cat_fc = nn.Linear(1024, 512)
        self.cosine_classifier = CosineClassifier(512, num_classes, scale=10.0)
        self.out_fc = nn.Linear(512, num_classes)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, abcd_features):
        cnn_features = self.cnn(x)
        cnn_features = self.flatten(cnn_features)
        cnn_features = torch.relu(self.cnn_fc(cnn_features))

        if abcd_features.dim() == 1:
            abcd_features = abcd_features.unsqueeze(0)
        transformed_abcd = torch.relu(self.abcd_fc(abcd_features))

        # Объединение через конкатенацию
        combined = self.alpha * cnn_features + (1 - self.alpha) * transformed_abcd  # [batch, 512]
        combined = torch.relu(combined)
        return self.cosine_classifier(combined)

    def apply_sparsity(self, sparsity_rate=0.5):
        """
        Применяет динамическую разреженность.
        sparsity_rate: процент обнулённых весов
        """
        with torch.no_grad():
            # Применяем обрезку к каждому слою Linear
            for name, param in self.named_parameters():
                if "weight" in name:  # Только для весов
                    abs_weight = torch.abs(param.data)
                    threshold = torch.quantile(abs_weight, sparsity_rate)  # Порог для обрезки
                    param.data[abs_weight < threshold] = 0  # Обнуляем веса ниже порога
