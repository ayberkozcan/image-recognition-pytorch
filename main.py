if __name__ == "__main__":
    import sys
    
    import torch
    from torch import nn

    import torchvision
    import matplotlib.pyplot as plt

    from src import prepare_data, freeze_layers, train_model, save_model 

    train_dir = "data/pets/train"
    test_dir = "data/pets/test"
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    print(torch.cuda.is_available())
    sys.exit()
    train_dataloader, test_dataloader, class_names = prepare_data.prepare_data(
        train_dir,
        test_dir,
    )

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    model_eff0 = freeze_layers.freeze_layers(model, class_names, device)

    results_eff0, total_time_eff0 = train_model.train_model(
        model_eff0,
        0.001,
        10,
        train_dataloader,
        test_dataloader,
        device
    )

    print(results_eff0)