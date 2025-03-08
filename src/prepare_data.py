from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def prepare_data(root, split, transform, batch_size):
    transform = transforms.Compose([
        transforms.Resize((423, 640)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = datasets.OxfordIIITPet(
        root,
        split,
        transform=transform
    )
    test_data = datasets.OxfordIIITPet(
        root,
        split,
        transform=transform
    )

    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )

    return train_dataloader, test_dataloader, class_names