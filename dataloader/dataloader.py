# dataloader/dataloader.py
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import DataLoader as TorchDataLoader, random_split,Subset
from torchvision.transforms import functional as F
import torch

class PetDatasetLoader:
    """DataLoader for Oxford-IIIT Pet Dataset (PyTorch version)"""

    @staticmethod
    def get_transforms(image_size):
        """Returns transform pipeline"""
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    @staticmethod
    def load_data(data_config):
        transform = PetDatasetLoader.get_transforms(data_config.image_size)

        def target_transform(mask):
            resized = F.resize(mask, (data_config.image_size, data_config.image_size), interpolation=F.InterpolationMode.NEAREST)
            return torch.as_tensor(F.pil_to_tensor(resized), dtype=torch.int64).squeeze()

        dataset = OxfordIIITPet(
            root=data_config.path,
            download=True,
            target_types="segmentation",
            transform=transform,
            target_transform=target_transform
        )
        tiny_subset = Subset(dataset, range(20))  # <-- only first 20 images
        return tiny_subset
        #return dataset

    @staticmethod
    def preprocess_data(dataset, batch_size, val_split=0.2):
        """Splits into train/test sets and returns DataLoaders"""
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size

        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = TorchDataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_ds, batch_size=batch_size)

        return train_loader, val_loader
