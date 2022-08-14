import warnings
from typing import List, Literal

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from dissector.mid_layer_extractor import get_mid_layer_extractor
from dissector.constants import ORIGINAL_OUTPUT


class SubModel(pl.LightningModule):
    def __init__(self, orig_model, middle_feat_dict: dict[str, str], feat_seq: List[str], n_classes: int = 1000, lr: float = 0.001):
        super(SubModel, self).__init__()
        self.save_hyperparameters(ignore=['orig_model', 'middle_feat_dict'])
        self.orig_model = get_mid_layer_extractor(orig_model, middle_feat_dict)
        self.orig_model.requires_grad_(False)
        classifiers = {
            v: nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(n_classes)
            )
            for v in feat_seq if v != ORIGINAL_OUTPUT
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

    def metrics(self, y, y_hat, feat_name, stage):
        acc = self.acc(y_hat, y)
        loss = self.loss(y_hat, y)
        self.log(f'{feat_name}_{stage}_acc', acc, sync_dist=True)
        self.log(f'{feat_name}_{stage}_loss', loss, sync_dist=True)
        return acc, loss

    def training_step(self, batch, batch_idx):
        opts = self.optimizers(use_pl_optimizer=True)
        x, y = batch
        outputs = self(x)
        y_hat = outputs[ORIGINAL_OUTPUT]
        _ = self.metrics(y, y_hat, 'origin', 'train')
        for idx, k in enumerate(self.classifiers.keys()):
            y_hat = outputs[k]
            _, loss = self.metrics(y, y_hat, k, 'train')
            opts[idx].zero_grad()
            self.manual_backward(loss)
            opts[idx].step()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        for k, v in outputs.items():
            _ = self.metrics(y, v, k, 'val')

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        for k, v in outputs.items():
            _ = self.metrics(y, v, k, 'test')
        pv = self.pv_score(batch)
        self.log(f'pv', pv.mean(), sync_dist=True)

    def sv_score(self, batch):
        x = batch if isinstance(batch, torch.Tensor) else batch[0]
        outputs = self(x)
        lx = torch.topk(outputs[ORIGINAL_OUTPUT], 1).indices[:, 0]  # [B, 1], highest prob label
        sv = []
        for k in self.classifiers.keys():
            # top labels and probs
            probs = torch.nn.functional.softmax(outputs[k], dim=1)
            top2 = torch.topk(probs, k=2)
            top2_prob = top2.values
            top_label = top2.indices[:, 0]
            lx_prob = probs[torch.arange(0, probs.shape[0], dtype=torch.long), lx]
            # cal sv score
            correct_mask = top_label == lx
            correct_sv = top2_prob[:, 0] / (top2_prob[:, 0] + top2_prob[:, 1])
            wrong_sv = 1. - top2_prob[:, 0] / (top2_prob[:, 0] + lx_prob)
            sv.append(correct_mask * correct_sv + ~correct_mask * wrong_sv)
        return torch.stack(sv)

    def pv_score(self, batch, growth_type: Literal['linear1', 'leaner2', 'log1', 'exp1'] = 'log1'):
        sv = self.sv_score(batch)
        x = torch.arange(1, len(self.classifiers) + 1, device=self.device)
        match growth_type:
            case 'linear1':
                weights = x
            case 'linear2':
                weights = 100 * x + 1
            case 'log1':
                weights = torch.log(x + 1)
            case 'exp1':
                weights = torch.exp(x)
            case _:
                warnings.warn(f'growth_type {growth_type} is not supported, using `linear1` as fallback')
                weights = x
        return torch.sum(sv.T * weights, dim=1) / torch.sum(weights)
