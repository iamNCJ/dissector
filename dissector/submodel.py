import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from dissector.mid_layer_extractor import get_mid_layer_extractor


class SubModel(pl.LightningModule):
    def __init__(self, orig_model, middle_feat_dict, n_classes: int = 1000, lr: float = 0.001):
        super(SubModel, self).__init__()
        self.save_hyperparameters(ignore=['orig_model', 'middle_feat_dict'])
        self.orig_model = get_mid_layer_extractor(orig_model, middle_feat_dict)
        self.orig_model.requires_grad_(False)
        classifiers = {
            v: nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(n_classes)
            )
            for v in middle_feat_dict.values() if v != 'original_output'
        }
        self.classifiers = nn.ModuleDict(classifiers)
        self.loss = torch.nn.CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy()
        self.automatic_optimization = False

    def forward(self, x):
        x = self.orig_model(x)
        for k, v in self.classifiers.items():
            x[k] = v(x[k])
        return x

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(classifier.parameters(), lr=self.hparams.lr)
            for classifier in self.classifiers.values()
        ]  # ModuleDict is **ordered**
        return optimizers, []

    def training_step(self, batch, batch_idx):
        opts = self.optimizers(use_pl_optimizer=True)
        x, y = batch
        outputs = self(x)
        y_hat = outputs['original_output']
        acc = self.acc(y_hat, y)
        self.log(f'origin_train_acc', acc)
        loss = self.loss(y_hat, y)
        self.log(f'origin_train_loss', loss)
        for idx, k in enumerate(self.classifiers.keys()):
            y_hat = outputs[k]
            acc = self.acc(y_hat, y)
            self.log(f'{k}_train_acc', acc)
            loss = self.loss(y_hat, y)
            self.log(f'{k}_train_loss', loss)
            opts[idx].zero_grad()
            self.manual_backward(loss)
            opts[idx].step()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        for k, v in outputs.items():
            if k != 'original_output':
                acc = self.acc(v, y)
                self.log(f'{k}_val_acc', acc)
                loss = self.loss(v, y)
                self.log(f'{k}_val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        for k, v in outputs.items():
            if k != 'original_output':
                acc = self.acc(v, y)
                self.log(f'{k}_test_acc', acc)
                loss = self.loss(v, y)
                self.log(f'{k}_test_loss', loss)


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
