

import torch
from torchvision.datasets import CIFAR10
from pytorch_lightning.callbacks import LearningRateMonitor

from pytorch_lightning.callbacks import TQDMProgressBar


import flash
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageEmbedder

# 1. Download the data and prepare the datamodule
# datamodule = ImageClassificationData.from_datasets(
#    train_dataset=CIFAR10("/workspace/DATA", download=False),
#    batch_size=8,
#)


from torchvision import transforms, datasets


import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for asashis example
from torchvision.datasets import MNIST
from torchvision import transforms


class DRACDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", transformer: str = "./",):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transformer
        
    def prepare_data(self):
        print('Already downloaded')
        # download
#         MNIST(self.data_dir, train=True, download=True)
#         MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            data_full = datasets.ImageFolder(root=self.data_dir)
            print(len(data_full))
            self.data_train, self.data_val = random_split(data_full, [511, 100])

        # Assign test dataset for use in dataloader(s)
        #if stage == "test" or stage is None:
        #    self.data_test = MNIST(self.data_dir, train=False, transform=self.transform)

        #if stage == "predict" or stage is None:
       #     self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=256, transforms=self.transform)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=256, transforms=self.transform)

#     def test_dataloader(self):
#         return DataLoader(self.mnist_test, batch_size=32)

#     def predict_dataloader(self):
#         return DataLoader(self.mnist_predict, batch_size=32)











data_transform = transforms.Compose([
        transforms.RandomCrop(512),
         transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
    ])





#drac_dataset = datasets.ImageFolder(root='/workspace/DATA/C. Diabetic Retinopathy Grading',transform=data_transform)

#dataset_loader = torch.utils.data.DataLoader(drac_dataset,
#                                             batch_size=4, shuffle=True,
#                                             num_workers=1)


datamodule = DRACDataModule(data_dir='/workspace/DATA/C. Diabetic Retinopathy Grading', transformer=data_transform)

#ImageClassificationData.from_datasets(
#    train_dataset=CIFAR10("/workspace/DATA", download=False),
#    batch_size=8,
#)

#print(datamodule)


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
trainer = flash.Trainer(strategy="ddp",max_epochs=1, gpus=1,
    callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)])
trainer.fit(embedder, datamodule=datamodule)

# 4. Save the model!
trainer.save_checkpoint("image_embedder_simclr_drac.pt")

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

