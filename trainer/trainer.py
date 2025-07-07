# trainer/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def train(self):
        self.model.train()
        num_epochs = 1  # Just one epoch for testing

        for epoch in range(num_epochs):
            running_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")

            for images, masks in pbar:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

            print(f"[Epoch {epoch+1}] Total Loss: {running_loss:.4f}")
