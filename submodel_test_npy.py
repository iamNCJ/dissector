from data import NPYDataModule
from models import ResNet20CIFAR10
from dissector import SubModel
import pytorch_lightning as pl


if __name__ == '__main__':
    orig_model = ResNet20CIFAR10()
    feat_dict = {
        'relu': 'feat_0',
        'layer1.2.relu': 'feat_1',
        'layer2.2.relu': 'feat_2',
        'layer3.2.relu': 'feat_3',
    }
    feat_seq = ['feat_0', 'feat_1', 'feat_2', 'feat_3']
    sub_model = SubModel.load_from_checkpoint(
        './checkpoints/sub_model.ckpt',
        orig_model=orig_model,
        middle_feat_dict=feat_dict,
        feat_seq=feat_seq
    )
    trainer = pl.Trainer(accelerator='auto')
    for name in ['BE', 'BIM', 'DeepFool', 'Dropout-SIA', 'FGSM', 'PGD', 'SIA', 'Dropout-SIA-4', 'Dropout-SIA-4-Targeted', 'PGD-Targeted', 'SIA-Targeted']:
        print(name)
        dm = NPYDataModule(image_npy_file=f'../SIA-research/ae/cifar10/{name}/ae.npy',
                           label_npy_file=f'../SIA-research/ae/cifar10/{name}/label.npy')
        trainer.test(sub_model, dm)
