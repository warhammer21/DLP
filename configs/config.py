# configs/config.py
CFG = {
    "data": {
        "path": "./data",  # Local path for datasets
        "image_size": 128
    },
    "train": {
        "batch_size": 64,
        "buffer_size": 1000,  # Ignored in PyTorch but left for compatibility
        "epoches": 20,
        "val_subsplits": 5,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "input": [3, 128, 128],  # PyTorch uses CHW format
        "up_stack": {
            "layer_1": 512,
            "layer_2": 256,
            "layer_3": 128,
            "layer_4": 64,
            "kernels": 3
        },
        "output": 3
    }
}
