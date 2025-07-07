from utils.config import Config
from dataloader.dataloader import PetDatasetLoader
from configs.config import CFG
from model.unet import UNet
from trainer.trainer import Trainer

def main():
    config = Config.from_dict(CFG)

    config.train.batch_size = 4  # override to 4

    dataset = PetDatasetLoader.load_data(config.data)
    print(dataset)
    print('xx' * 14)

    train_loader, val_loader = PetDatasetLoader.preprocess_data(dataset, config.train.batch_size)

    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        break

    model = UNet(config)  # PyTorch nn.Module

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    trainer.train()

if __name__ == "__main__":
    main()
