import torch
from pydantic import DirectoryPath, FilePath
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MAIN_DIR: DirectoryPath = "./data/all_data/"
    PATH_TO_LABELS_CSV: FilePath = "./data/all_data/labels.csv"

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


settings = Settings()


def get_settings():
    return settings
