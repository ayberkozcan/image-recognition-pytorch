import torch

def freeze_layers(model, class_names, device):
    for param in model.features.parameters():
        param.requires_grad = False

    torch.manual_seed(42)
    
    in_shape = model.classifier[1].in_features
    output_shape = len(class_names)

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(
            in_features=in_shape,
            out_features=output_shape,
            bias=True
        )
    ).to(device)