"""Example fully connected network architecture."""

import torch


class FullyConnected(torch.nn.Sequential):
    def __init__(self, in_dim, out_dim, hidden_dim, hidden_lay):
        layers = [torch.nn.Linear(in_dim, hidden_dim)]
        layers.append(torch.nn.ReLU())
        for _ in range(hidden_lay - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_dim, out_dim))
        super().__init__(*layers)

class FCNAugmented(torch.nn.Sequential):
    def __init__(self, in_dim, out_dim, hidden_dim, hidden_lay):
        output = torch.nn.Softmax(dim=-1) if out_dim > 1 else torch.nn.Sigmoid()
        layers = [torch.nn.Flatten(), torch.nn.Linear(in_dim, hidden_dim)]
        layers.append(torch.nn.ReLU())
        for _ in range(hidden_lay - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_dim, out_dim))
        layers.append(output)
        super().__init__(*layers)
