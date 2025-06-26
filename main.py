from utils.config import Config
from dataloader.dataloader import PetDatasetLoader
from configs.config import CFG

def main():
    config = Config.from_dict(CFG)

    dataset = PetDatasetLoader.load_data(config.data)
    train_loader, val_loader = PetDatasetLoader.preprocess_data(dataset, config.train.batch_size)

    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        break

if __name__ == "__main__":
    main()
