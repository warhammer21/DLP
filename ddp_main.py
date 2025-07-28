import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import copy
from utils.config import Config
from dataloader.dataloader import PetDatasetLoader
from configs.config import CFG
from model.unet import UNet
from trainer.trainer import Trainer

def average_models(models):
    avg_model = copy.deepcopy(models[0])
    for key in avg_model.state_dict().keys():
        avg_param = torch.stack([m.state_dict()[key].float() for m in models], dim=0).mean(dim=0)
        avg_model.state_dict()[key].copy_(avg_param)
    return avg_model

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def ddp_main(rank, world_size):
    setup(rank, world_size)

    config = Config.from_dict(CFG)
    config.train.batch_size = 4

    dataset = PetDatasetLoader.load_data(config.data)
    train_loader, val_loader = PetDatasetLoader.preprocess_data(dataset, config.train.batch_size)

    # Split dataset manually (simulate different workers)
    total_len = len(train_loader.dataset)
    part_len = total_len // world_size
    indices = list(range(total_len))
    local_indices = indices[rank * part_len: (rank + 1) * part_len]

    local_subset = torch.utils.data.Subset(train_loader.dataset, local_indices)
    local_loader = torch.utils.data.DataLoader(local_subset, batch_size=4, shuffle=True)

    model = UNet(config)

    trainer = Trainer(model=model, train_loader=local_loader, val_loader=val_loader, config=config)
    trainer.train()

    # Save model from each process
    torch.save(model.state_dict(), f"model_rank_{rank}.pt")

    cleanup()

def run():
    world_size = 2  # Simulate 2 processes
    mp.spawn(ddp_main, args=(world_size,), nprocs=world_size, join=True)

    # After training, average saved models
    models = []
    for i in range(2):
        model = UNet(Config.from_dict(CFG))
        model.load_state_dict(torch.load(f"model_rank_{i}.pt"))
        models.append(model)

    final_model = average_models(models)
    print("Final averaged model:")
    print(final_model)

if __name__ == "__main__":
    run()
