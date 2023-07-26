import yaml
import itertools

# Base configuration dictionary
base_config = {
    "dataset": {
        "root_folder": "/home/dim26fa/data/shapenet",
        "classes": ["all"],
        "suffix": ".pts"
    },
    "model": "fold",
    "train": {
        "lr": 0.0001,
        "batch_size": 16,
        "num_workers": 4,
        "momentum": 0.9,
        "momentum2": 0.999,
        "num_epochs": 1000,
        "cd_loss": "ChamferDistanceL2",
        "early_stop_patience": 300,
        "log_dir": "/home/dim26fa/coding/training",
        "scheduler_type": None,
        "gamma": 1e-6
    },
    "pcn_config": {
        "coarse_dim": 16384,
        "fine_dim": 1024
    },
    "adapointr_config": {
        "NAME": 'AdaPoinTr',
        "num_query": 512,
        "num_points": 16384,
        "center_num": [512, 256],
        "global_feature_dim": 1024,
        "encoder_type": 'graph',
        "decoder_type": 'fc',
        "encoder_config": {
            "embed_dim": 384,
            "depth": 6,
            "num_heads": 6,
            "k": 8,
            "n_group": 2,
            "mlp_ratio": 2.0,
            "block_style_list": ['attn-graph', 'attn', 'attn', 'attn', 'attn', 'attn'],
            "combine_style": 'concat'},
        "decoder_config": {
            "embed_dim": 384,
            "depth": 8,
            "num_heads": 6,
            "k": 8,
            "n_group": 2,
            "mlp_ratio": 2.0,
            "self_attn_block_style_list": ['attn-graph', 'attn', 'attn', 'attn', 'attn', 'attn', 'attn', 'attn'],
            "self_attn_combine_style": 'concat',
            "cross_attn_block_style_list": ['attn-graph', 'attn', 'attn', 'attn', 'attn', 'attn', 'attn', 'attn'],
            "cross_attn_combine_style": 'concat'}
        }
    }


# The varying parameters
batch_sizes = list(range(8, 129, 8))
models = ['fold', 'pointr', 'pcn']
schedulers = [None, True]
learning_rates = [0.01, 0.001, 0.0001]

def generate_config_files(config, batch_sizes, models, schedulers, learning_rates):
    # Iterate over all combinations of the varying parameters
    for batch_size, model, scheduler, lr in itertools.product(batch_sizes, models, schedulers, learning_rates):
        config['train']['batch_size'] = batch_size
        config['model'] = model
        config['train']['scheduler_type'] = scheduler
        config['train']['lr'] = lr

        # Define the file name based on the parameters
        filename = f"config_{model}_bs{batch_size}_lr{lr}_sch{scheduler}.yaml"

        # Write to the file
        with open(filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

# Generate the config files
generate_config_files(base_config, batch_sizes, models, schedulers, learning_rates)
