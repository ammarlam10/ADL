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
trainer = flash.Trainer(strategy="ddp",max_epochs=1, gpus=1,
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