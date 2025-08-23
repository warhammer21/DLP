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
    print(f"[Rank {rank}] Initializing process")

    setup(rank, world_size, args.backend)

    config = Config.from_dict(CFG)
    config.train.batch_size = args.batch_size

    # Load dataset
    dataset = PetDatasetLoader.load_data(config.data)
    train_loader, val_loader = PetDatasetLoader.preprocess_data(dataset, config.train.batch_size)

    total_train_images = len(train_loader.dataset)
    print(f"[Rank {rank}] Total training images: {total_train_images}")

    # Manual split for this simple DDP version
    part_len = total_train_images // world_size
    indices = list(range(total_train_images))
    local_indices = indices[rank * part_len: (rank + 1) * part_len]

    print(f"[Rank {rank}] Will train on {len(local_indices)} images: {local_indices[:5]}...")

    local_subset = torch.utils.data.Subset(train_loader.dataset, local_indices)
    local_loader = torch.utils.data.DataLoader(local_subset, batch_size=args.batch_size, shuffle=True)

    # Show how many batches per rank
    num_batches = len(local_loader)
    print(f"[Rank {rank}] Number of batches: {num_batches} with batch size {args.batch_size}")

    model = UNet(config)
    trainer = Trainer(model=model, train_loader=local_loader, val_loader=val_loader, config=config)

    print(f"[Rank {rank}] Starting training...")
    trainer.train()
    print(f"[Rank {rank}] Finished training.")

    torch.save(model.state_dict(), f"model_rank_{rank}.pt")
    print(f"[Rank {rank}] Model saved to model_rank_{rank}.pt")

    cleanup()


def run(args):
    world_size = args.world_size
    print(f"Launching training with {world_size} processes")
    mp.spawn(ddp_main, args=(world_size, args), nprocs=world_size, join=True)

    print("All processes finished. Now averaging models...")
    models = []
    for i in range(world_size):
        model = UNet(Config.from_dict(CFG))
        model.load_state_dict(torch.load(f"model_rank_{i}.pt"))
        models.append(model)
        print(f"Loaded model from model_rank_{i}.pt")

    final_model = average_models(models)
    print("Final averaged model ready.")
    print(final_model)


def parse_args():
    parser = argparse.ArgumentParser(description="DDP Trainer")
    parser.add_argument('--world_size', type=int, default=2, help='Number of processes to simulate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per process')
    parser.add_argument('--backend', type=str, default='gloo', help='Distributed backend')
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
