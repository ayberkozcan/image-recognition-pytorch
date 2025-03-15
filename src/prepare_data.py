import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def prepare_data(train_dir, test_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = datasets.OxfordIIITPet(
        root=train_dir,
        split="trainval",
        download=True,
        transform=transform
    )

    test_data = datasets.OxfordIIITPet(
        root=test_dir,
        split="test",
        download=True,
        transform=transform
    )

    print("\n[INFO] Dataset is ready\n")

    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        num_workers=2,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    print("\n[INFO] Dataloaders are ready\n")

    return train_dataloader, test_dataloader, class_names
