from torchvision import transforms


def get_transformer():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ],
    )
