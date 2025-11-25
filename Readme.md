### PhD Project _"Towards enhanced particle averaging for single-molecule localization microscopy using geometric deep learning"_

This repository contains scripts, notebooks and helper functions for the "Towards enhanced particle averaging for single-molecule localization microscopy using geometric deep learning".  

| [Installation](#installation) | [Training](#training) | [Inference](#inference) | [Tutorial](#tutorial-notebooks) | [Data simulation](#data-simulation) | [Container](#containerized-simulation-application) | [Contents](#contents) |
***
### Installation 

Using mamba package manager, one can create a new environment for this repository and install the dependencies by running  

`mamba create --name <env_name>`

`mamba activate <env_name>`

`pip install .` in the smlm folder 

For installing Chamfer distance and pointnet2_ops: 
`git clone https://github.com/qinglew/PCN-PyTorch`
`cd PCN-PyTorch/extensions/chamfer`
`pip install .`

`git clone https://github.com/fishbotics/pointnet2_ops.git`
`cd pointnet2_ops`
`pip install .`

1. Clone the original PCN repository (contains the custom CUDA ops that this project reuses):
   ```
   git clone https://github.com/qinglew/PCN-PyTorch.git
   ```
   Make sure you have a recent CUDA toolkit build system available in your environment before proceeding.

2. Build and install the Chamfer Distance extension used by PCN:
   ```
   cd PCN-PyTorch/chamfer_distance
   pip install .
   ```

3. *(Optional)* Install the PointNet++ ops if you plan to use modules that depend on them (e.g. custom FPS utilities):
   ```
   git clone https://github.com/fishbotics/pointnet2_ops.git
   cd pointnet2_ops
   pip install .
   ```

The `pip install .` command will compile the CUDA operators against the currently active environment and register them so they can be imported from this repository.

***

### Training
To train PocaFoldAS on your data or the bundled demo data:

1. Ensure dependencies are installed (see Installation).
2. Make sure `train.log_dir` in your config points to a writable directory for checkpoints/logs. When running in a container, mount the host folder you want to use (e.g., `-v /host/logs:/workspace/smlm_logs`) and set `train.log_dir` to that container path instead of relying on notebook-only env vars.
3. Run: `python scripts/run_training.py --config configs/config_demo_data.yaml --exp_name demo_run`.
   - Uses `demo_data/tetrahedron_seed1121_train` by default via the config.
   - Logs/checkpoints go to the `log_dir` set in the config (override via env / config copy).
   - Set `train.use_wandb: true` in the config or toggle `USE_WANDB` in the training notebook; the notebook writes a config copy to `artifacts/notebook_configs/` respecting `TRAIN_LOG_DIR`.

### Inference
For script-based inference (no notebook):
1. Update the `test` section of your config (e.g., `configs/test_config.yaml`) so `ckpt_path` points to the trained weights and `log_dir` points to a writable output directory. Mount that folder when running in a container (e.g., `-v /host/infer:/workspace/smlm_inference`) and set `test.log_dir` to the container path.
2. Ensure the dataset/test settings in the config reflect the split you want to evaluate (defaults use `demo_data/tetrahedron_seed1234_test`).
3. Run `python scripts/test_pocafoldas.py --config configs/test_config.yaml`. Metrics and exported clouds are written under `test.log_dir`.

For notebooks we still expose env vars for convenience:
- Set `POCAFOLDAS_CKPT` to the checkpoint (default `weights/tetra.pth`).
- Set `POCAFOLDAS_INFER_OUT` if you need a different output directory (default `/workspace/smlm_inference`).
- Use `tutorial/Inference_and_visualization.ipynb`, which respects those env vars.

### Tutorial notebooks

We created tutorial notebooks for an easy understanding of the project. 
* Introduction and data simulation  <a target="_blank" href="https://colab.research.google.com/github/dianamindroc/smlm/blob/master/tutorial/Intro_and_data_simulation.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

* Network training <a target="_blank" href="https://colab.research.google.com/github/dianamindroc/smlm/blob/master/tutorial/Network_training.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

* Inference and visualization <a target="_blank" href="https://colab.research.google.com/github/dianamindroc/smlm/blob/master/tutorial/Inference_and_visualization.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

#### Running notebooks in a container
To run the notebooks from the container we created and avoid setting up all the enviroment from scratch, follow these steps: 
- Pull the published image with Singularity if you prefer it over Docker: `singularity pull pocafoldas_latest.sif docker://dianamindroc/pocafoldas:latest`, then `singularity exec pocafoldas_latest.sif bash`
- Mount a writable log/checkpoint folder: `-v /host/logs:/workspace/smlm_logs` (training outputs) and optionally `-v /host/infer:/workspace/smlm_inference` (inference outputs)
- Map the notebook port from the container to the host when you start it (e.g., `-p 8888:8888` for Docker, or pass `--port 8888 --ip 0.0.0.0` to `jupyter lab` inside Singularity) so you can open the `http://127.0.0.1:<port>` in your browser.
- After you start the container shell, export the env vars the notebooks read: `TRAIN_LOG_DIR=/workspace/smlm_logs` for training logs, `POCAFOLDAS_CKPT` for the checkpoint path, and `POCAFOLDAS_INFER_OUT` if you override the inference output directory
- Launch Jupyter/Colab inside the container from the repo root

***

### Data simulation
There are three ways to simulate data:

#### 1. Simulation using SuReSim
To use the simulation code, the original java code of the SureSim simulator from [Kuner Lab](https://github.com/tkunerlab/JavaUmsetzungSTORMSimulation) needs to be built. 
Subsequently, the path to the .jar file (the target of the project) will be used in the configuration file `config.yaml`. 
To run the simulations from the command line, navigate to the scripts folder and run simulations:

`python run_simulation.py --config config.yaml`

Default number of simulated samples is 15. The default microscopy technique used for simulation parameters is dSTORM. Epitope density, recorded frames and detection efficiency are the simulation parameters that are randomly varied in specific ranges. To modify these ranges, navigate to `simulate_data.py` in `simulation` folder.

#### 2. Simulation using script 
To simulate using in-house script, go to scripts and run the `simulate_data.py` script e.g.
`cd scripts`

`python simulate_data.py -s 'cube' -n 10 -rot True` where -s is structure desired, choose from cube, pyramid, tetrahedron, sphere; -n is number of samples to simulate and -rot is whether to rotate the model structure or not. Additional prompts will be displayed in the terminal. 

#### 3. Containerized simulation application
 **_NOTE_** Prerequisites: [Singularity](https://sylabs.io/guides/3.0/user-guide/quick_start.html) installed on the host machine.

The simulator is containerized in a Singularity container and can be obtained by building the container from the definition file provided in `/container/simulation.def` with the command 

`singularity build simulator_container.sif simulation.def`

The container expects a folder `model` in the working directory which contains the ground truth model to simulate samples from. Accepted formats are .wimp and .txt. 
Afterwards, to run a simulation from the container, run the command 

`singularity run simulator_container.sif 10 dstorm` , where 10 is the number of desired simulated samples and dstorm is the microscopy technique (options: dstorm or palm).

If another folder is to be used for models, the container can be run in interactive mode: 

`singularity shell simulator_container.sif`

Edit the config file accordingly and run `python3 /smlm/scripts/run_simulation.py --config new_config.yaml` inside the container with default 15 samples and dstorm technique or `smlm/scripts/run_simulation.py --config new_config.yaml -s number_of_samples -t technique`

***

### Contents

```
smlm/
|-- Readme.md
|-- setup.py
|-- pyproject.toml
|-- requirements.txt
|-- Dockerfile
|-- configs/                # YAML configs that drive training/inference experiments
|   |-- config_demo_data.yaml
|   |-- config_adapointr_bs8_lr0.01_schNone_test.yaml
|   |-- config.yaml
|   `-- test_config.yaml
|-- container/              # Singularity definitions for reproducible training/simulation
|   |-- training.def
|   `-- simulation.def
|-- dataset/                # Dataset wrappers and loaders
|   |-- load_dataset.py
|   |-- Dataset.py
|   |-- ShapeNet.py
|   |-- SMLMDataset.py
|   `-- SMLMSimulator.py
|-- demo_data/              # Small demo splits used by the configs
|   |-- tetrahedron_seed1121_train/
|   `-- tetrahedron_seed1234_test/
|-- helpers/                # Shared utilities for data IO, logging, stats, config helpers
|   |-- data.py
|   |-- logging.py
|   |-- pc_stats.py
|   |-- readers.py
|   |-- visualization.py
|   `-- generate_config_files.py
|-- model_architectures/    # Network definitions and loss layers
|   |-- pocafoldas.py
|   |-- folding_net.py
|   |-- pcn_decoder.py
|   |-- adaptive_folding.py
|   |-- losses.py
|   `-- transforms.py
|-- notebooks/              # Research notebooks for experiments and visualization
|   |-- Graph Autoencoder.ipynb
|   |-- PointNet.ipynb
|   |-- smlm_preprocessing.ipynb
|   `-- visualization.ipynb
|-- preprocessing/          # Point-cloud preprocessing CLI entry
|   `-- preprocessing.py
|-- resources/              # Static helper assets (paths, lookups)
|   |-- highest_shape.csv
|   `-- paths.txt
|-- scripts/                # Command-line entry points and utilities
|   |-- run_training.py
|   |-- test_pocafoldas.py
|   |-- run_simulation.py
|   |-- simulate_data.py
|   |-- run_preprocessing.py
|   `-- npc_smlm_averaging.py
|-- simulation/             # Higher-level simulation workflows
|   |-- simulate_suresim_data.py
|   |-- simulate_with_custom_model.py
|   `-- suresim_simulator.py
|-- tutorial/               # Hosted notebooks referenced in the README
|   |-- Intro_and_data_simulation.ipynb
|   |-- Network_training.ipynb
|   `-- Inference_and_visualization.ipynb
|-- chamfer_distance/       # CUDA chamfer distance op used by the models
|   |-- chamfer_distance.cpp
|   |-- chamfer_distance.cu
|   `-- chamfer_distance.py
|-- weights/                # Reference checkpoints distributed with the repo
|   |-- tetra.pth
|   `-- npc.pth
|-- submodules/
|   `-- dgcnn/              # External dependency for point-set baselines
`-- figures/               # Static images used in papers/docs
```
