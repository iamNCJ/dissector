import torch
from torch.fx import symbolic_trace

from dissector.constants import ORIGINAL_OUTPUT


def get_mid_layer_extractor(module: torch.nn.Module, middle_feat_dict: dict[str, str]):
    proxy_output_dict = {}
    symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
    for node in symbolic_traced.graph.nodes:
        if node.target in middle_feat_dict.keys():
            proxy_output_dict[middle_feat_dict[node.target]] = node
        if node.op == 'output':
            proxy_output_dict[ORIGINAL_OUTPUT] = node.args[0]  # output node only has one arg
            node.update_arg(0, proxy_output_dict)

    symbolic_traced.graph.lint()
    return torch.fx.GraphModule(module, symbolic_traced.graph)


if __name__ == '__main__':
    from models.resnet_timm import ResNet50
    model = ResNet50(device='mps')
    mock_input = torch.randn((1, 3, 299, 299), device='mps')
    middle_feat_dict = {
        'model.1.act1': 'feat_0',
        'model.1.layer1.2.act3': 'feat_1',
        'model.1.layer2.3.act3': 'feat_2',
        'model.1.layer3.5.act3': 'feat_3',
        'model.1.layer4.2.act3': 'feat_4'
    }
    new_model = get_mid_layer_extractor(model, middle_feat_dict)
    x = new_model(mock_input)
    for k, v in x.items():
        print(k, end=' ')
        print(v.shape)
    print(x[ORIGINAL_OUTPUT])
