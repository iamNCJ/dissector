import torch
import torch.nn as nn

from dissector.mid_layer_extractor import get_mid_layer_extractor


class SubModel(nn.Module):
    def __init__(self, orig_model, middle_feat_dict, n_classes: int = 1000):
        super(SubModel, self).__init__()
        self.orig_model = get_mid_layer_extractor(orig_model, middle_feat_dict)
        classifiers = {
            v: nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(n_classes)
            )
            for v in middle_feat_dict.values() if v != 'original_output'
        }
        self.classifiers = nn.ModuleDict(classifiers)

    def forward(self, x):
        x = self.orig_model(x)
        for k, v in self.classifiers.items():
            x[k] = v(x[k])
        return x


if __name__ == '__main__':
    from models import ResNet50
    device = 'mps'
    model = ResNet50(device=device)
    middle_feat_dict = {
        'model.1.act1': 'feat_0',
        'model.1.layer1.2.act3': 'feat_1',
        'model.1.layer2.3.act3': 'feat_2',
        'model.1.layer3.5.act3': 'feat_3',
        'model.1.layer4.2.act3': 'feat_4'
    }
    sub_model = SubModel(model, middle_feat_dict)
    sub_model = sub_model.to(device)
    mock_input = torch.randn((50, 3, 299, 299), device=device)
    x = sub_model(mock_input)
    for _k, _v in x.items():
        print(_k, end=' ')
        print(_v.shape)
