import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def prepare_data(train_dir, test_dir):
    transform = transforms.Compose([
        transforms.Resize((423, 640)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = datasets.OxfordIIITPet(
        train_dir,
        split="trainval",
        transform=transform,
        download=True
    )

    test_data = datasets.OxfordIIITPet(
        test_dir,
        split="test",
        transform=transform,
        download=True
    )

    print("\n[INFO] Dataset is ready\n")

    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        num_workers=1,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=32,
        shuffle=False,
        num_workers=1
    )
    
    print("\n[INFO] Dataloaders are ready\n")

    return train_dataloader, test_dataloader, class_names