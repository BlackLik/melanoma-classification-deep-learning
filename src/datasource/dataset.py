from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.datasource.abcd import compute_abcd_features
from src.utils import to_scalar


class SkinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: Path, transform: transforms.Compose | None = None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_class = str(self.df.iloc[idx, 0])
        img_name = self.df.iloc[idx, 1]
        img_path = Path(self.root_dir, img_class, img_name)

        if not img_path.exists():
            print(f"⚠️ Файл не найден: {img_path}")
            return None  # Пропускаем отсутствующие файлы

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"⚠️ Ошибка загрузки изображения: {img_path}")

            return None  # Пропускаем пустые изображения

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        abcd_features_image = image.copy()
        abcd_features = compute_abcd_features(abcd_features_image)
        abcd_tensor = torch.tensor(
            [to_scalar(value) for value in abcd_features.values()],
            dtype=torch.float,
        )

        if self.transform:
            image = self.transform(image)

        data = {
            "malignant": 1,
            "benign": 0,
        }

        label = torch.tensor(data.get(self.df.iloc[idx, 0], 2), dtype=torch.long)

        return image, label, abcd_tensor
