from data import CIFAR10DataModule
from models import ResNet20CIFAR10
from dissector import SubModel
import pytorch_lightning as pl


if __name__ == '__main__':
    dm = CIFAR10DataModule('./data/cifar10')
    orig_model = ResNet20CIFAR10()
    orig_model.load_weights('./resnet20-12fca82f.th')
    feat_dict = {
        'bn1': 'feat_0',
        'layer1.2.shortcut': 'feat_1',
        'layer2.2.shortcut': 'feat_2',
        'layer3.2.shortcut': 'feat_3',
    }
    sub_model = SubModel(dm, feat_dict, n_classes=10)
    trainer = pl.Trainer(gpus=-1)
    trainer.fit(sub_model, dm)
    trainer.test(sub_model, dm)
    trainer.save_checkpoint('./checkpoints/sub_model.ckpt')
