import numpy as np
import torch


def to_scalar(x) -> float:
    # Если это тензор PyTorch
    if isinstance(x, torch.Tensor):
        # Если содержит ровно один элемент, то берем .item()
        if x.numel() == 1:
            return float(x.item())

        return float(x.mean().item())

    # Если это массив numpy, делаем тоже самое
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return float(x)

        return float(x.mean())

    return float(x)
