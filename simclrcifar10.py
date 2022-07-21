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
'''train_transforms = torchvision.transforms.Compose(
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
'''
#cifar10_dm = CIFAR10DataModule(
#    data_dir=PATH_DATASETS,
#    batch_size=BATCH_SIZE,
#    num_workers=NUM_WORKERS,
#    train_transforms=train_transforms,
#    test_transforms=test_transforms,
#    val_transforms=test_transforms,
#)


# DATASET transform=train_transforms,
cifar10_data_train = torchvision.datasets.CIFAR10('../',  train=True,download=True)

# train loader
train_data_loader = torch.utils.data.DataLoader(cifar10_data_train,
                                          batch_size=512,
                                          shuffle=True,
                                          num_workers=4)
 #transform=test_transforms
cifar10_data_test = torchvision.datasets.CIFAR10('../', train=False,download=True)
test_data_loader = torch.utils.data.DataLoader(cifar10_data_test,
                                          batch_size=512,
                                          shuffle=True,
                                          num_workers=4)

print('Train length',len(train_data_loader.dataset))
#exit()

# train_dataset = MyDataset(transforms=SimCLRTrainDataTransform())
# val_dataset = MyDataset(transforms=SimCLREvalDataTransform())

# simclr needs a lot of compute!
model = SimCLR(max_epochs=100,num_samples=len(train_data_loader.dataset), batch_size=512, gpus=1,dataset='cifar10')



# trainer = Trainer(devices=1, accelerator="gpu",callbacks=[TQDMProgressBar(refresh_rate=10)])
trainer = Trainer(devices=1, accelerator="gpu")
print('TRAINER')

trainer.fit(
    model,
    train_data_loader,
)



# Additional information
EPOCH = 10
PATH = "./cifar10model.pt"

torch.save({'model_state_dict': model.state_dict()}, PATH)