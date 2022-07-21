import torch
from torchvision.datasets import CIFAR10

import flash
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageEmbedder

# 1. Download the data and prepare the datamodule
datamodule = ImageClassificationData.from_datasets(
    train_dataset=CIFAR10(".", download=True),
    batch_size=8,
)

# 2. Build the task
embedder = ImageEmbedder(
    backbone="resnet18",
    training_strategy="barlow_twins",
    head="barlow_twins_head",
    pretraining_transform="barlow_twins_transform",
    training_strategy_kwargs={"latent_embedding_dim": 128},
    pretraining_transform_kwargs={"size_crops": [32]},
)


model = embedder.load_from_checkpoint("image_embedder_simclr.pt")


# 3. Create the trainer and pre-train the encoder
trainer = flash.Trainer(max_epochs=1, gpus=torch.cuda.device_count())
#trainer.fit(embedder, datamodule=datamodule)

# 4. Save the model!

embeddings = trainer.predict(model, datamodule=datamodule)

# list of embeddings for images sent to the predict function
print(embeddings)