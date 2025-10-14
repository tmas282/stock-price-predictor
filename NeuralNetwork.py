from torch import nn
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.non_linear_Sequential = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        pass
    def forward(self, x):
        y = nn.non_linear_Sequential
        return y