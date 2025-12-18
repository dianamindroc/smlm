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
2. (Optional) Set envs for container paths: `TRAIN_LOG_DIR` (default `/workspace/smlm_logs`).
3. Run: `python scripts/run_training.py --config configs/config_demo_data.yaml --exp_name demo_run`.
   - Uses `demo_data/tetrahedron_seed1121_train` by default via the config.
   - Logs/checkpoints go to the `log_dir` set in the config (override via env / config copy).
   - Set `train.use_wandb: true` in the config or toggle in the training notebook to log to Weights & Biases.

### Inference
Use the demo test split and the provided tetra checkpoint (stored under `weights/`):

- Point `POCAFOLDAS_CKPT` to the demo tetra checkpoint, e.g., `weights/pocafoldas_tetra_demo.pth` (adjust to actual filename).
- If running in a container, set `POCAFOLDAS_INFER_OUT` to a writable/mounted directory (default `/workspace/smlm_inference`).
- Run inference:
  - Notebook: `tutorial/Inference_and_visualization.ipynb` (respects the env vars above).
  - Or via script: `python scripts/test_pocafoldas.py --config configs/config_demo_data.yaml --ckpt $POCAFOLDAS_CKPT` (if you add a CLI wrapper).

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
To simplify environment requirements, we built a docker container of the repository. 
- Mount a writable log/checkpoint folder: `-v /host/logs:/workspace/smlm_logs` (training outputs) and optionally `-v /host/infer:/workspace/smlm_inference` (inference outputs).
- Set env vars for the notebooks: `TRAIN_LOG_DIR=/workspace/smlm_logs` for training, `POCAFOLDAS_CKPT` for a checkpoint, and `POCAFOLDAS_INFER_OUT` for inference outputs if you override the defaults.
- Launch Jupyter/Colab inside the container from the repo root; the notebooks already respect these env vars and will write configs to `artifacts/notebook_configs/`.



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
smlm
│   README.md  
│   setup.py
│   .gitignore
│   LICENSE.txt
│   requirements.txt
└───configs
│   │   config.yaml
│   │   test_config.yaml
└───container
│   │   config.yaml
│   │   training.def
│   │   simulation.def
└───dataset
│   │   load_dataset.py
│   │   ShapeNet.py
│   │   SMLMDataset.py
│   │   SMLMSimulator.py
└───helpers
│   │   chamfer.py
│   │   data.py
│   │   logging.py
│   │   pc_stats.py
│   │   readers.py
│   │   visualization.py
└───model_architectures
│   │   adaptive_folding.py
│   │   attention.py
│   │   chamfer_distance_updated.py
│   │   chamfer_distances.py
│   │   folding_net.py
│   │   losses.py
│   │   pocafoldas.py
│   │   pcn_decoder.py
│   │   transforms.py
│   │   utils.py
└───notebooks
│   │   visualization.ipynb
│   │   smlm_preprocessing.ipynb 
└───preprocessing
│   │   preprocessing.py
└───scripts
│   │   run_simulation.py
│   │   config.yaml
│   │   main.py
│   │   run_preprocessing.py
└───simulation
│   │   simulator.py
│   │   simulate_data.py
└───tutorial
│   │   Intro_and_data_simulation.ipynb
└──────  
```
