import timm
import torch
from torch import nn
from torchvision import models

from .cosine_classifier import CosineClassifier


class ResNetVitModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Используем resnet18 без последнего fc-слоя
        resnet = models.resnet18(pretrained=True)
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)  # Загрузим предобученную модель ViT
        self.vit.head = nn.Identity()  # Убираем последний слой (классификатор), чтобы получить признаки
        self.vit_fc = nn.Linear(768, 512)  # Слои для преобразования признаков ViT

        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.cnn_fc = nn.Linear(512, 512)

        # Вместо стандартного выходного слоя используем CosineClassifier,
        # который принимает объединённые признаки размерности 512
        self.cat_fc = nn.Linear(1024, 512)
        self.cosine_classifier = CosineClassifier(512, num_classes, scale=10.0)

        self.out_fc = nn.Linear(512, num_classes)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x):
        cnn_features = self.cnn(x)
        cnn_features = self.flatten(cnn_features)
        cnn_features = torch.relu(self.cnn_fc(cnn_features))

        vit_features = self.vit(x)  # Признаки из ViT
        vit_features = vit_features.view(vit_features.size(0), -1)  # Приводим размерность
        vit_features = torch.relu(self.vit_fc(vit_features))

        combined = self.alpha * cnn_features + (1 - self.alpha) * vit_features
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
                    # Порог для обрезки
                    threshold = torch.quantile(abs_weight, sparsity_rate)
                    # Обнуляем веса ниже порога
                    param.data[abs_weight < threshold] = 0
