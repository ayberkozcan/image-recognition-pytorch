import torch
from torch import nn

def train_model(model, learning_rate, epoch_number, train_dataloader, test_dataloader, device):

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

  def accuracy_fn(y_true, y_pred):
    correct = (y_true == y_pred).sum().item()
    return (correct / len(y_true)) * 100

  torch.manual_seed(42)
  torch.cuda.manual_seed(42)

  from tqdm.auto import tqdm

  from timeit import default_timer as timer
  train_time_start = timer()

  epochs = epoch_number

  results = {
      "train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }

  for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n--------")

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(train_dataloader):
      X, y = X.to(device), y.to(device)

      model.train()

      y_pred = model(X)

      loss = loss_fn(y_pred, y)
      train_loss += loss

      train_acc += accuracy_fn(
          y_true=y,
          y_pred=y_pred.argmax(dim=1)
      )

      optimizer.zero_grad()

      loss.backward()

      optimizer.step()

      if batch % 400 == 0:
        print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    test_loss, test_acc = 0, 0

    model.eval()
    with torch.inference_mode():
      for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        test_pred = model(X)

        loss = loss_fn(test_pred, y)
        test_loss += loss

        accuracy = accuracy_fn(
            y_true=y, y_pred=test_pred.argmax(dim=1)
        )
        test_acc += accuracy

      test_loss /= len(test_dataloader)
      test_acc /= len(test_dataloader)

    print(f"\nTrain Loss: {train_loss:.5f} | Train Accuracy: {train_acc:.2f}%, Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}%\n")

    results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
    results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
    results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
    results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

  train_time_end = timer()
  total_time = train_time_end - train_time_start

  print(f"Total Time: {total_time} seconds")

  return results, total_time