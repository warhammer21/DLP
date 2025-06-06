# main.py

from utils.config import Config
from dataloader.dataloader import PetDatasetLoader

def main():
    # Load config
    config = Config.from_json("configs/cect_config.json")  # or however you access your CFG

    # Load and preprocess dataset
    dataset = PetDatasetLoader.load_data(config.data)
    train_loader, val_loader = PetDatasetLoader.preprocess_data(dataset, config.train.batch_size)

    # Test: print a single batch shape
    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        break

if __name__ == "__main__":
    main()
