from data import CIFAR10DataModule
from models import ResNet20CIFAR10
from dissector import SubModel
import pytorch_lightning as pl


if __name__ == '__main__':
    # init
    orig_model = ResNet20CIFAR10()
    feat_dict = {
        'relu': 'feat_0',
        'layer1.2.relu': 'feat_1',
        'layer2.2.relu': 'feat_2',
        'layer3.2.relu': 'feat_3',
    }
    sub_model = SubModel.load_from_checkpoint(
        './checkpoints/sub_model.ckpt',
        orig_model=orig_model,
        middle_feat_dict=feat_dict
    )
    print(sub_model)
    # trainer = pl.Trainer()
    dm = CIFAR10DataModule('./data/cifar10')
    # trainer.test(sub_model, dm)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    sub_model.sv_score(batch)
