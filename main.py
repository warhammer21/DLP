from utils.config import Config
from dataloader.dataloader import PetDatasetLoader
from configs.config import CFG

def main():
    config = Config.from_dict(CFG)

    dataset = PetDatasetLoader.load_data(config.data)
    print(dataset)
    print('xx'*14)
    train_loader, val_loader = PetDatasetLoader.preprocess_data(dataset, config.train.batch_size)

    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        break

if __name__ == "__main__":
    main()


# main.py

# from configs.config import CFG
# from dataloader.dataloader import PetDatasetLoader
# #from model.unet import UNet
#
# def run():
#     config = CFG
#
#     # Step 1: Load data
#     data_loader = PetDatasetLoader()
#     dataset, info = data_loader.load_data(config["data"])
#     train_ds, test_ds = data_loader.preprocess_data(
#         dataset=dataset,
#         batch_size=config["train"]["batch_size"],
#         buffer_size=config["train"]["buffer_size"],
#         image_size=config["data"]["image_size"]
#     )
#
#     # # Step 2: Initialize and build model
#     # model = UNet(config)
#     # model.build()
#     #
#     # # Step 3: Train model
#     # model.train(train_ds, test_ds)
#     #
#     # # Step 4: Evaluate model
#     # model.evaluate(test_ds)
#
# if __name__ == "__main__":
#     run()
