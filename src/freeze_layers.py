import torch
import torchvision.models

def freeze_layers(model, class_names, device):
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "features") and hasattr(model, "classifier"):
        # For EfficientNet
        in_shape = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=in_shape, out_features=len(class_names), bias=True)
        )

    elif hasattr(model, "fc"):
        # For ResNet
        in_shape = model.fc.in_features
        model.fc = torch.nn.Linear(in_features=in_shape, out_features=len(class_names), bias=True)

    else:
        raise TypeError("Unsupported model type. Only EfficientNet and ResNet are supported.")

    model.to(device)
    return model
