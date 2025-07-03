# trainer/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim

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
        for epoch in range(self.config.train.epoches):
            running_loss = 0.0
            for images, masks in self.train_loader:
                # Forward
                outputs = self.model(images)

                loss = self.criterion(outputs, masks)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch+1}/{self.config.train.epoches}], Loss: {running_loss:.4f}")

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            for images, masks in self.val_loader:
                outputs = self.model(images)
                # You can add accuracy, IoU, Dice, etc.
                print(f"Evaluated batch outputs shape: {outputs.shape}")
                break  # Just one batch for sanity check
