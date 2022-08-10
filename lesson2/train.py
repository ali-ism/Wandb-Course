import wandb
from ml_collections import config_dict
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, ToTensor, Compose
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet18, mobilenet_v3_small, densenet121
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics import F1Score
from sklearn.utils.class_weight import compute_class_weight
from pytorch_lightning.loggers import WandbLogger


PROJECT_NAME = 'garbage-project'
RAW_DATA_FOLDER = '../input/garbage-classification/Garbage classification/Garbage classification/'

#default hyperparameters
cfg = config_dict.ConfigDict()
cfg.seed = 1
cfg.img_size = 224
cfg.batch_size = 32
cfg.lr = 0.0001
cfg.arch = 'resnet'
cfg.epochs = 5
cfg.fc_neurons = 128


def load_data(cfg):
    transforms = Compose([Resize((cfg.img_size,cfg.img_size)), ToTensor()])
    dataset = ImageFolder(RAW_DATA_FOLDER, transform=transforms)
    dataset.idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    train_indices, test_indices, y_train, _ = train_test_split(
        range(len(dataset)),
        dataset.targets,
        stratify=dataset.targets,
        test_size=0.1,
        random_state=cfg.seed
    )
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    train_indices, val_indices, _, _ = train_test_split(
        train_dataset.indices,
        y_train,
        stratify=y_train,
        test_size=0.111,
        random_state=cfg.seed
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return dataset, train_dataset, val_dataset, test_dataset


class Model(pl.LightningModule):

    def __init__(self, cfg, dataset):
        super().__init__()
        self.cfg = cfg
        num_classes = len(set(dataset.targets))
        self.f1_score = F1Score(num_classes=num_classes, average='macro')
        self.class_weights = torch.Tensor(compute_class_weight(class_weight='balanced',
                                                               classes=np.unique(dataset.targets),
                                                               y=dataset.targets))
        
        if cfg.arch == 'resnet':
            self.net = resnet18('DEFAULT')
            self.net.fc = nn.Linear(512, cfg.fc_neurons)
        elif cfg.arch == 'mobilenet':
            self.net = mobilenet_v3_small('DEFAULT')
            self.net.classifier[-1] = nn.Linear(1024, cfg.fc_neurons)
        elif cfg.arch == 'densenet':
            self.net = densenet121('DEFAULT')
            self.net.classifier = nn.Linear(1024, cfg.fc_neurons)
        else:
            raise ValueError("Architecture should be either 'resnet', 'mobilenet' or 'densenet'.")
        
        self.out = nn.Linear(cfg.fc_neurons, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(x)
        x = self.out(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y, weight=self.class_weights.to(y_hat.device))
        f1 = self.f1_score(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_f1', f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y, weight=self.class_weights.to(y_hat.device))
        f1 = self.f1_score(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y, weight=self.class_weights.to(y_hat.device))
        f1 = self.f1_score(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.forward(x)
        return x, y, y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr)

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.cfg.batch_size, num_workers=2)

    def val_dataloader(self):
        return DataLoader(val_dataset, batch_size=self.cfg.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(test_dataset, batch_size=self.cfg.batch_size, num_workers=2)

      
def train(cfg):
    pl.seed_everything(seed=cfg.seed, workers=True)
    wandb_logger = WandbLogger(job_type='train-sweep')
    dataset, train_dataset, val_dataset, test_dataset = load_data(cfg)
    model = Model(cfg, dataset)
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator='auto',
        deterministic=True,
        logger=wandb_logger)
    trainer.fit(model)


if __name__ == '__main__':
    train(cfg)
