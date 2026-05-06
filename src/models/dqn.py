import torch
from torch import nn

class DQN(nn.Module):
    '''
    2015 Nature DQN
    '''
    def __init__(self, n_actions: int) -> None:
        super().__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=32,
            kernel_size=8,
            stride=4,
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
        )

        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, 4, 84, 84)
            n_flatten = self.flatten(self.conv3(self.conv2(self.conv1(dummy)))).size(1)

        self.fc1 = nn.Linear(in_features=n_flatten, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
