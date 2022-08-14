from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

import pytorch_lightning as pl


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size: int = 32, split_ratio: float = 0.8, num_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.cifar_train = None
        self.cifar_val = None
        self.cifar_test = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cifar_full = datasets.CIFAR10(self.data_dir, train=True, transform=self.transform, download=True)
            cifar_len = len(cifar_full)
            train_len = int(cifar_len * self.split_ratio)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [train_len, cifar_len - train_len])
        if stage == "test" or stage is None:
            self.cifar_test = datasets.CIFAR10(self.data_dir, train=False, transform=self.transform, download=True)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == '__main__':
    dm = CIFAR10DataModule('./cifar10')
    dm.setup()
    ld = dm.train_dataloader()
    x, y = next(iter(ld))
    print(x.shape, y.shape)
