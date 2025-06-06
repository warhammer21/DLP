# model/unet.py
import tensorflow as tf
from dataloader.dataloader import DataLoader
from utils.logger import get_logger
from .base_model import BaseModel

LOG = get_logger('unet')

class UNet(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.dataset = None
        self.info = None
        self.train_dataset = None
        self.test_dataset = None
        self.batch_size = self.config.train.batch_size
        self.buffer_size = self.config.train.buffer_size
        self.image_size = self.config.data.image_size

    def load_data(self):
        LOG.info(f'Loading {self.config.data.path} dataset...')
        self.dataset, self.info = DataLoader.load_data(self.config.data)
        self.train_dataset, self.test_dataset = DataLoader.preprocess_data(
            self.dataset, self.batch_size, self.buffer_size, self.image_size
        )
