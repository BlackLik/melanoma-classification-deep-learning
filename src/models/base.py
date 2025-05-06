import logging
from timeit import default_timer

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from src import config
from src.datasource.base import SkinDataset

settings = config.get_settings()


class PyTorchModel:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler=None,
        l1_lambda: float = 1e-5,
        *,
        verbose: bool = True,
        use_abcd_test: bool = True,
    ):
        """
        Инициализация класса PyTorchModel.

        model: nn.Module — модель PyTorch
        criterion: nn.Module — функция потерь (например, nn.CrossEntropyLoss)
        optimizer: optim.Optimizer — оптимизатор (например, optim.Adam)
        scheduler: torch.optim.lr_scheduler — (необязательно) планировщик для изменения скорости обучения
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_abcd_test = use_abcd_test
        self.l1_lambda = l1_lambda

        self.train_acc_history = []
        self.val_acc_history = []
        self.loss_history = []

        logger = logging.getLogger()
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO if verbose else logging.WARNING)

        self.logger = logger

    def l1_regularization(self):
        l1_norm = sum(p.abs().sum() for p in self.model.parameters())
        return self.l1_lambda * l1_norm

    def fit(
        self,
        train_dataloader: DataLoader[SkinDataset],
        validation_dataloader: DataLoader[SkinDataset],
        epochs=10,
        device=settings.DEVICE,
    ) -> None:
        start_time = default_timer()
        self.model.to(device=device)

        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, total = 0, 0, 0

            for batch in train_dataloader:
                if batch is None:
                    continue
                    
                self.optimizer.zero_grad()
                image, label, abcd_features = batch
                image, label, abcd_features = image.to(device), label.to(device), abcd_features.to(device)
                args: tuple[Tensor] = (image, abcd_features) if self.use_abcd_test else (image,)
                outputs: Tensor = self.model(*args)
                loss = self.criterion(outputs, label)
                l1_loss = self.l1_regularization()
                all_loss = loss + l1_loss

                all_loss.backward()
                self.optimizer.step()

                total_loss += all_loss.item()
                correct += (outputs.argmax(dim=1) == label).sum().item()
                total += label.size(0)

            self.loss_history.append(total_loss)
            train_acc = correct / total
            self.train_acc_history.append(train_acc)

            if hasattr(self.model, "apply_sparsity"):
                self.model.apply_sparsity()

            # Оценка на валидации
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in validation_dataloader:
                    image, label, abcd_features = batch
                    image, label, abcd_features = image.to(device), label.to(device), abcd_features.to(device)
                    args: tuple[Tensor] = (image, abcd_features) if self.use_abcd_test else (image,)
                    outputs: Tensor = self.model(*args)
                    correct += (outputs.argmax(dim=1) == label).sum().item()
                    total += label.size(0)

            val_acc = correct / total
            self.val_acc_history.append(val_acc)

            self.logger.info(
                "Epoch %s / %s  - Loss: %s, Train Acc: %s, Val Acc: %s",
                epoch + 1,
                epochs,
                round(total_loss, 4),
                round(train_acc, 4),
                round(val_acc, 4),
            )
            if self.scheduler:
                self.scheduler.step()

        stop_time = default_timer() - start_time
        hours = stop_time // 3600
        minutes = (stop_time % 3600) // 60
        seconds = stop_time % 60
        self.logger.info(
            "Training time: %s:%s:%s seconds",
            int(hours),
            int(minutes),
            round(seconds, 2),
        )

    def evaluate_on_test_f1(
        self,
        test_loader: DataLoader[SkinDataset],
        device=settings.DEVICE,
    ):
        self.model.eval()  # Переключаем модель в режим оценки
        all_labels = []
        all_preds = []

        with torch.no_grad():  # Отключаем вычисление градиентов
            for batch in test_loader:
                image, label, abcd_features = batch
                image, label, abcd_features = image.to(device), label.to(device), abcd_features.to(device)
                args: tuple[Tensor] = (image, abcd_features) if self.use_abcd_test else (image,)
                outputs: Tensor = self.model(*args)
                _, predicted = outputs.max(1)  # Получаем предсказания модели

                all_labels.extend(label.cpu().numpy())  # Собираем истинные метки
                # Собираем предсказания модели
                all_preds.extend(predicted.cpu().numpy())

        # Вычисляем F1-меру
        # Средневзвешенная F1-метрика
        return f1_score(all_labels, all_preds, average="weighted")

    def plot_training_and_validation_accuracy(self):
        plt.plot(range(len(self.train_acc_history)), self.train_acc_history, label="Train Accuracy")
        plt.plot(range(len(self.val_acc_history)), self.val_acc_history, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Training and Validation Accuracy")

    def plot_loss(self):
        plt.plot(range(len(self.loss_history)), self.loss_history, label="Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss score")
        plt.legend()
        plt.title("Loss History")

    def load_model(self, path_to_weights: str):
        """
        Загружает веса в модель.
    
        :param path_to_weights: Путь к файлу с сохранёнными весами .pth
        """
        state_dict = torch.load(path_to_weights, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Переводим в режим inference
        print(f"✅ Модель загружена из {path_to_weights}!")

