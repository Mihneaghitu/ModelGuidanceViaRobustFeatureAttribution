import torch

class LesionNet(torch.nn.Sequential):
    def __init__(self, in_channels, out_dim):
        output = torch.nn.Softmax(dim=-1) if out_dim > 1 else torch.nn.Sigmoid()
        super().__init__(
            torch.nn.Conv2d(in_channels, 16, 4, 2, 0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 4, 4, 0),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(43808, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, out_dim),
            output,
        )

class PlantNet(torch.nn.Sequential):
    def __init__(self, in_channels, out_dim):
        output = torch.nn.Softmax(dim=-1) if out_dim > 1 else torch.nn.Sigmoid()
        super().__init__(
            torch.nn.Conv2d(in_channels, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(173056, 1024, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, out_dim),
            output,
        )

class DermaNet(torch.nn.Sequential):
    def __init__(self, in_channels, feature_size, out_dim):
        self.latent_dim = {28: 2304, 64: 14400, 128: 61504, 224: 193600}
        super().__init__(
            torch.nn.Conv2d(in_channels, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(self.latent_dim[feature_size], 1024, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, out_dim),
            torch.nn.Sigmoid()
        )


class SalientImageNet(torch.nn.Sequential):
    def __init__(self, in_channels, out_dim):
        output = torch.nn.Softmax(dim=-1) if out_dim > 1 else torch.nn.Sigmoid()
        super().__init__(
            torch.nn.Conv2d(in_channels, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(173056, 1024, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, out_dim),
            output
        )
