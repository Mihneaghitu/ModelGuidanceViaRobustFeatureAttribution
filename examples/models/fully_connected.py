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
    def __init__(self, in_dim, out_dim, hidden_dim, hidden_lay, init_with_small_weights=False):
        output = torch.nn.Softmax(dim=-1) if out_dim > 1 else torch.nn.Sigmoid()
        layers = [torch.nn.Flatten(), torch.nn.Linear(in_dim, hidden_dim)]
        layers.append(torch.nn.ReLU())
        for _ in range(hidden_lay - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_dim, out_dim))
        layers.append(output)
        super().__init__(*layers)
        if init_with_small_weights:
            self.__init_with_small_weights()

    def __init_with_small_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=0.01)
                torch.nn.init.constant_(m.bias, 0)
