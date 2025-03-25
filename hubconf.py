# hubconf.py
from torchvision import models


def mel_cdl(*, pretrained=True):
    return models.resnet18(pretrained=pretrained)
