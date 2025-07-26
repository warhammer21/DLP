from utils.config import Config
from dataloader.dataloader import PetDatasetLoader
from configs.config import CFG
from model.unet import UNet
from trainer.trainer import Trainer
import copy
import torch

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
    train_loader1, train_loader2 = torch.utils.data.random_split(train_loader.dataset, [len(train_loader.dataset)//2]*2)
    loader1 = torch.utils.data.DataLoader(train_loader1, batch_size=4, shuffle=True)
    loader2 = torch.utils.data.DataLoader(train_loader2, batch_size=4, shuffle=True)

    model = UNet(config)
    model1 = UNet(config)
    model2 = UNet(config)

# PyTorch nn.Module

    def average_models(models):
        avg_model = copy.deepcopy(models[0])
        for key in avg_model.state_dict().keys():
            avg_param = torch.stack([m.state_dict()[key].float() for m in models], dim=0).mean(dim=0)
            avg_model.state_dict()[key].copy_(avg_param)
        return avg_model


    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    #trainer.train()
    # Trainer is reused for each model
    trainer1 = Trainer(model=model1, train_loader=loader1, val_loader=val_loader, config=config)
    trainer2 = Trainer(model=model2, train_loader=loader2, val_loader=val_loader, config=config)

    # Train each model on its own "worker"
    trainer1.train()
    trainer2.train()

    final_model = average_models([model1, model2])
    print(final_model)





if __name__ == "__main__":
    main()
