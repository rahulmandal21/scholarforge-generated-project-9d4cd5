import torch
import torch.nn as nn

class BinaryCrossEntropyLoss(nn.Module):
    """
    A PyTorch implementation of binary cross-entropy loss function.
    """

    def __init__(self) -> None:
        """
        Initialize the binary cross-entropy loss function.
        """
        super().__init__()
        self.criterion = nn.BCELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the binary cross-entropy loss between predictions and targets.

        Args:
        predictions (torch.Tensor): The predicted probabilities.
        targets (torch.Tensor): The true labels.

        Returns:
        torch.Tensor: The binary cross-entropy loss.
        """
        return self.criterion(predictions, targets)


class ModelTrainer:
    """
    A class to train a PyTorch model using binary cross-entropy loss and gradient descent optimization.
    """

    def __init__(self, model: nn.Module, device: torch.device, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """
        Initialize the model trainer.

        Args:
        model (nn.Module): The PyTorch model to train.
        device (torch.device): The device to use for training.
        loss_fn (nn.Module): The loss function to use.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        """
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_one_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """
        Train the model for one epoch.

        Args:
        dataloader (torch.utils.data.DataLoader): The data loader to use.

        Returns:
        float: The average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)


if __name__ == "__main__":
    # Create a dummy model
    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(5, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.fc(x))

    # Create a dummy data loader
    dummy_data = torch.randn(100, 5)
    dummy_targets = torch.randint(0, 2, (100, 1)).float()
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(dummy_data, dummy_targets), batch_size=10)

    # Create a model trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DummyModel()
    loss_fn = BinaryCrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = ModelTrainer(model, device, loss_fn, optimizer)

    # Train the model for one epoch
    loss = trainer.train_one_epoch(dataloader)
    print(f"Loss: {loss}")