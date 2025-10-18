import torch
from torch import nn
from torch import optim

loss_fn = nn.MSELoss()

class StockPriceNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.non_linear_Sequential = nn.Sequential(
            nn.Linear(30 * 6, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        pass
    def flatten(self, x):
        f = nn.Flatten()
        return f(x)
    def forward(self, x):
        y = self.non_linear_Sequential(x)
        return y

    def train_loop(self, dataloader: torch.utils.data.DataLoader, batch_size=64, learning_rate=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        size = len(dataloader)
        self.train()
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = self(self.flatten(X))
            pred = torch.squeeze(pred)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    def test_loop(self, dataloader: torch.utils.data.DataLoader):
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss= 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, y in dataloader:
                pred = self(self.flatten(X))
                test_loss += loss_fn(pred, y).item()

        test_loss /= num_batches
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")