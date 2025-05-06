from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src import config
from src.datasource.dataset import SkinDataset
from src.datasource.transformer import get_transformer

def _check_file(row: dict) -> bool:
    settings = config.get_settings()
    return Path(settings.MAIN_DIR, row["class"], row["filename"]).exists()

def get_data_frame():
    settings = config.get_settings()
    df_data = pd.read_csv(settings.PATH_TO_LABELS_CSV)
    df_data = df_data[df_data["class"] != "SkinCancer"]
    df_data = df_data[df_data["class"] != "Moles"]
    df_data = df_data[df_data["class"] != "Unknown_Normal"]
    df_data = df_data[df_data.apply(_check_file, axis=1)]

    return df_data.sample(frac=1.0, random_state=42).reset_index(drop=True)


def get_train_validate_test(df_data) -> tuple[SkinDataset, SkinDataset, SkinDataset]:
    train_val, test = train_test_split(
        df_data,
        test_size=0.1,
        random_state=42,
        stratify=df_data["class"],
    )
    train, val = train_test_split(train_val, test_size=0.1, random_state=42, stratify=train_val["class"])

    transform = get_transformer()
    settings = config.get_settings()

    train_dataset = SkinDataset(df=train, root_dir=settings.MAIN_DIR, transform=transform)
    val_dataset = SkinDataset(df=val, root_dir=settings.MAIN_DIR, transform=transform)
    test_dataset = SkinDataset(df=test, root_dir=settings.MAIN_DIR, transform=transform)

    return train_dataset, val_dataset, test_dataset


def get_data_loader(dataset: SkinDataset):
    return DataLoader(dataset=dataset, batch_size=16, shuffle=True)


def build_tabular_data(dataset: SkinDataset):
    data = []
    for sample in dataset:
        if sample is None:
            continue
        _, label, abcd_features = sample

        # Если label – тензор, преобразуем в число
        label_val = label.item() if isinstance(label, torch.Tensor) else label
        # Предполагаем, что abcd_features – это словарь с ключами
        data.append(
            {
                "asymmetry": abcd_features[0].item(),
                "border_irregularity": abcd_features[1].item(),
                "color_variation": abcd_features[2].item(),
                "diameter": abcd_features[3].item(),
                "abcd_score": abcd_features[4].item(),
                "label": label_val,
            },
        )

    return pd.DataFrame(data)
