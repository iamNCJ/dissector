from data import CIFAR10DataModule
from models import ResNet20CIFAR10
from dissector import SubModel
import pytorch_lightning as pl


if __name__ == '__main__':
    dm = CIFAR10DataModule('./data/cifar10')
    orig_model = ResNet20CIFAR10()
    orig_model.load_weights('./resnet20-12fca82f.th')
    feat_dict = {
        'relu': 'feat_0',
        'layer1.2.relu': 'feat_1',
        'layer2.2.relu': 'feat_2',
        'layer3.2.relu': 'feat_3',
    }
    sub_model = SubModel(orig_model, feat_dict, n_classes=10)
    # dry run to initialize LazyModules
    dl = dm.train_dataloader()
    x, _ = next(iter(dl))
    sub_model(x)
    trainer = pl.Trainer(accelerator='auto')
    trainer.fit(sub_model, dm)
    trainer.test(sub_model, dm)
    trainer.save_checkpoint('./checkpoints/sub_model.ckpt')
