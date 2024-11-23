import torch
from torchvision.models import resnet18, ResNet18_Weights

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
    def __init__(self, in_channels, out_dim, gain=None):
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

        if gain is not None:
            self.init_weights(gain)

    def init_weights(self, gain):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=gain)
                torch.nn.init.constant_(m.bias, 0)

class DermaNet(torch.nn.Sequential):
    def __init__(self, in_channels, feature_size, out_dim, arch_type: str = "medium_large", init_with_small_weights: bool = False):
        self.latent_dim = {28: 2304, 64: 14400, 128: 61504, 224: 193600}
        self.in_channels = in_channels
        self.feature_size = feature_size
        self.out_dim = out_dim
        super().__init__(
            *self.__make_arch(arch_type)
        )
        if init_with_small_weights:
            self.__init_with_small_weights()

    def __init_with_small_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=0.1)
                torch.nn.init.constant_(m.bias, 0)


    def __make_arch(self, key: str) -> tuple[any]:
        match key:
            case "small":
                return (
                    torch.nn.Conv2d(self.in_channels, 32, 4, 1, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 32, 4, 2, 1),
                    torch.nn.Flatten(start_dim=1, end_dim=-1),
                    torch.nn.ReLU(),
                    torch.nn.Linear(30752, self.out_dim),
                    torch.nn.Sigmoid()
                )
            case "small_medium":
                return (
                    torch.nn.Conv2d(self.in_channels, 32, 4, 1, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 32, 4, 2, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 64, 4, 1, 1),
                    torch.nn.ReLU(),
                    torch.nn.Flatten(start_dim=1, end_dim=-1),
                    torch.nn.Linear(57600, 1024, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, self.out_dim, bias=True),
                    torch.nn.Sigmoid()
                )
            case "medium_large":
                return (
                    torch.nn.Conv2d(self.in_channels, 32, 3, 1, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 32, 4, 2, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 64, 4, 1, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 64, 4, 2, 1),
                    torch.nn.ReLU(),
                    torch.nn.Flatten(start_dim=1, end_dim=-1),
                    torch.nn.Linear(self.latent_dim[self.feature_size], 1024, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, 1024, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, self.out_dim),
                    torch.nn.Sigmoid()
                )
            case "large":
                return (
                    torch.nn.Conv2d(self.in_channels, 32, 3, 1, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 32, 4, 2, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 64, 4, 1, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 64, 4, 2, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 64, 4, 1, 1),
                    torch.nn.ReLU(),
                    torch.nn.Flatten(start_dim=1, end_dim=-1),
                    torch.nn.Linear(12544, 4096, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.Linear(4096, 1024, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, 128, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, self.out_dim),
                    torch.nn.Sigmoid()
                )


class SalientImageNet(torch.nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Identity()
        # freeze the resnet layer weights
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(num_features, num_classes)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        y = self.resnet(x)
        # y = self.dropout(y)
        y = self.fc(y)
        y = self.softmax(y)
        return y

class SeqSalientImageNet(torch.nn.Sequential):
    def __init__(self, modules: list[torch.nn.Module]):
        super().__init__(*modules)
