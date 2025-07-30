import os
import argparse
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


def setup(rank, world_size, backend):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def ddp_main(rank, world_size, args):
    setup(rank, world_size, args.backend)

    config = Config.from_dict(CFG)
    config.train.batch_size = args.batch_size

    dataset = PetDatasetLoader.load_data(config.data)
    train_loader, val_loader = PetDatasetLoader.preprocess_data(dataset, config.train.batch_size)

    # Split dataset manually
    total_len = len(train_loader.dataset)
    part_len = total_len // world_size
    indices = list(range(total_len))
    local_indices = indices[rank * part_len: (rank + 1) * part_len]

    local_subset = torch.utils.data.Subset(train_loader.dataset, local_indices)
    local_loader = torch.utils.data.DataLoader(local_subset, batch_size=args.batch_size, shuffle=True)

    model = UNet(config)
    trainer = Trainer(model=model, train_loader=local_loader, val_loader=val_loader, config=config)
    trainer.train()

    torch.save(model.state_dict(), f"model_rank_{rank}.pt")

    cleanup()


def run(args):
    world_size = args.world_size
    mp.spawn(ddp_main, args=(world_size, args), nprocs=world_size, join=True)

    # Average saved models
    models = []
    for i in range(world_size):
        model = UNet(Config.from_dict(CFG))
        model.load_state_dict(torch.load(f"model_rank_{i}.pt"))
        models.append(model)

    final_model = average_models(models)
    print("Final averaged model:")
    print(final_model)


def parse_args():
    parser = argparse.ArgumentParser(description="DDP Trainer")
    parser.add_argument('--world_size', type=int, default=2, help='Number of processes to simulate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per process')
    parser.add_argument('--backend', type=str, default='gloo', help='Distributed backend')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
