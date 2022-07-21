'''
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform



import os

import torch
#import torch.nn as nn
#import torch.nn.functional as F
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
#from pytorch_lightning.callbacks import LearningRateMonitor
#from pytorch_lightning.loggers import TensorBoardLogger
#from torch.optim.lr_scheduler import OneCycleLR
#from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

from pytorch_lightning.callbacks import TQDMProgressBar




seed_everything(7)
PATH_DATASETS = os.environ.get("./", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 512 if AVAIL_GPUS else 64

print(torch.cuda.device_count())
#exit()

NUM_WORKERS = int(os.cpu_count() / 2)


# define train transformer
train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

# Define test transformer

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform

cifar10_dm = CIFAR10DataModule(
    data_dir="/workspace/DATA",
    batch_size=512,
    num_workers=4,
    train_transforms=train_transforms,
    test_transforms=train_transforms,
    val_transforms=test_transforms,
)


it = iter(cifar10_dm)
first = next(it)
second = next(it)

print(len(first))
exit()


#cifar10_dm.train_transforms = train_transforms
#cifar10_dm.test_transforms = test_transforms

# DATASET transform=train_transforms,
#cifar10_data_train = torchvision.datasets.CIFAR10('../',  train=True,download=True)

# train loader
#train_data_loader = torch.utils.data.DataLoader(cifar10_data_train,
#                                          batch_size=512,
#                                          shuffle=True,
#                                          num_workers=4)
 #transform=test_transforms
#cifar10_data_test = torchvision.datasets.CIFAR10('../', train=False,download=True)
#test_data_loader = torch.utils.data.DataLoader(cifar10_data_test,
#                                          batch_size=512,
#                                          shuffle=True,
#                                          num_workers=4)


#print('Train length',len(train_data_loader.dataset))
#exit()

# train_dataset = MyDataset(transforms=SimCLRTrainDataTransform())
# val_dataset = MyDataset(transforms=SimCLREvalDataTransform())

# simclr needs a lot of compute!
# model = SimCLR(max_epochs=100,num_samples=len(train_data_loader.dataset), batch_size=512, gpus=1,dataset='cifar10')
model = SimCLR(max_epochs=100,num_samples=cifar10_dm.num_samples, batch_size=512, gpus=1,dataset='cifar10')



trainer = Trainer(devices=1, accelerator="gpu",callbacks=[TQDMProgressBar(refresh_rate=10)])
# trainer = Trainer(devices=1, accelerator="gpu")
print('TRAINER')

#trainer.fit(
#    model,
#    train_data_loader,
#)

trainer.fit(
    model,
    cifar10_dm,
)



# Additional information
EPOCH = 10
PATH = "./cifar10model.pt"

torch.save({'model_state_dict': model.state_dict()}, PATH)





import os

import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from IPython.core.display import display
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

seed_everything(7)

PATH_DATASETS = "/workspace/DATA"
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)

print(NUM_WORKERS)
print(PATH_DATASETS)


train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

cifar10_dm = CIFAR10DataModule(
    data_dir=PATH_DATASETS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)

def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model




class LitResnet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}




model = LitResnet(lr=0.05)

trainer =  (
    max_epochs=30,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    logger=CSVLogger(save_dir="logs/"),
    callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
)

trainer.fit(model, cifar10_dm)
trainer.test(model, datamodule=cifar10_dm)


'''

import torch
from torchvision.datasets import CIFAR10
from pytorch_lightning.callbacks import LearningRateMonitor

from pytorch_lightning.callbacks import TQDMProgressBar


import flash
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageEmbedder

# 1. Download the data and prepare the datamodule
datamodule = ImageClassificationData.from_datasets(
    train_dataset=CIFAR10("/workspace/DATA", download=False),
    batch_size=8,
)

# 2. Build the task
embedder = ImageEmbedder(
    backbone="resnet",
    training_strategy="simclr",
    head="simclr_head",
    pretraining_transform="simclr_transform",
    training_strategy_kwargs={"latent_embedding_dim": 128},
    pretraining_transform_kwargs={"size_crops": [32]},
)

# 3. Create the trainer and pre-train the encoder torch.cuda.device_count()
trainer = flash.Trainer(strategy="ddp",max_epochs=1, gpus=2,
    callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)])
trainer.fit(embedder, datamodule=datamodule)

# 4. Save the model!
trainer.save_checkpoint("image_embedder_simclr.pt")

# 5. Download the downstream prediction dataset and generate embeddings
#download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "data/")

#datamodule = ImageClassificationData.from_files(
#    predict_files=[
#        "data/hymenoptera_data/predict/153783656_85f9c3ac70.jpg",
#        "data/hymenoptera_data/predict/2039585088_c6f47c592e.jpg",
#    ],
#    batch_size=2,
#)
#embeddings = trainer.predict(embedder, datamodule=datamodule)

# list of embeddings for images sent to the predict function
#print(embeddings)