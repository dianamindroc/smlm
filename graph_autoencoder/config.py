"""
Model configuration
"""

cfg = {
    "data": {
        "path": ""
    },
    "train": {
        "batch_size": 64,
        "epochs": 200,
        "optimizer": {
            "type": "adam",
            "lr": 0.01
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "encoder": "GCNConv",
        "nr_features": 3,
        "output": 2
    }
}