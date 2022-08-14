import torch
from torch import nn
import timm


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


class ResNet(nn.Module):
    """
    ResNet Module that can extract features from the middle of the network
    """

    def __init__(self, resnet_type: str = 'resnet18', device: str = "cuda", dropout_rate: float = 0.05):
        super(ResNet, self).__init__()
        self.device = torch.device(device)
        self.force_dropout = False
        self.dropout_rate = dropout_rate
        norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.model = nn.Sequential(
            norm_layer,
            timm.create_model(resnet_type, pretrained=True)
        ).to(self.device)
        self.features = {}

    def hook_middle_representation(self):
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output

            return hook

        self.model[1].global_pool.register_forward_hook(get_features('feats_plr'))
        self.model[1].layer4.register_forward_hook(get_features('feats_4'))
        self.model[1].layer3.register_forward_hook(get_features('feats_3'))
        self.model[1].layer2.register_forward_hook(get_features('feats_2'))
        self.model[1].layer1.register_forward_hook(get_features('feats_1'))
        self.model[1].conv1.register_forward_hook(get_features('feats_0'))

    def hook_force_dropout(self):
        def force_dropout(module, input, output):
            return nn.functional.dropout(output, p=self.dropout_rate, training=self.force_dropout)

        self.eval()  # Monte Carlo dropout requires other parts of the models to be in eval mode, like BN
        for module in self.model[1].modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(force_dropout)

    def forward(self, x):
        return self.model(x.to(self.device))


class ResNet18(ResNet):
    def __init__(self, device: str = "cuda"):
        super(ResNet18, self).__init__(resnet_type='resnet18', device=device)


class ResNet50(ResNet):
    def __init__(self, device: str = "cuda"):
        super(ResNet50, self).__init__(resnet_type='resnet50', device=device)

