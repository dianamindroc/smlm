### PhD Project _"Towards enhanced particle for single-molecule localization microscopy using geometric deep learning"_

This repository contains scripts, notebooks and helper functions for the "Towards enhanced particle averaging for single-molecule localization microscopy using geometric deep learning".  

| [Installation](#installation) | [Training](#training) | [Inference](#inference) | [Tutorial](#tutorial-notebooks) | [Data simulation](#data-simulation) | [Container](#containerized-simulation-application) | [Contents](#contents) |

#### Installation 
Using mamba package manager, one can create a new environment for this repository and install the dependencies by running  

`mamba create --name <env_name>`

`pip install .` in the smlm folder 

#### Training

#### Inference

#### Tutorial notebooks
We created tutorial notebooks for an easy understanding of the project.
* Introduction and data simulation  <a target="_blank" href="https://colab.research.google.com/github/dianamindroc/smlm/blob/master/tutorial/Intro_and_data_simulation.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

#### Data simulation

To use the simulation code, the original java code of the SureSim simulator from [Kuner Lab](https://github.com/tkunerlab/JavaUmsetzungSTORMSimulation) needs to be built. 
Subsequently, the path to the .jar file (the target of the project) will be used in the configuration file `config.yaml`. 
To run the simulations from the command line, navigate to the scripts folder and run simulations:

`python run_simulation.py --config config.yaml`

Default number of simulated samples is 15. The default microscopy technique used for simulation parameters is dSTORM. Epitope density, recorded frames and detection efficiency are the simulation parameters that are randomly varied in specific ranges. To modify these ranges, navigate to `simulate_data.py` in `simulation` folder.

#### Containerized simulation application

Prerequisites: [Singularity](https://sylabs.io/guides/3.0/user-guide/quick_start.html) installed on the host machine.

The simulator is containerized in a Singularity container and can be obtained by building the container from the definition file provided in `/container/simulation.def` with the command 

`singularity build simulator_container.sif simulation.def`

The container expects a folder `model` in the working directory which contains the ground truth model to simulate samples from. Accepted formats are .wimp and .txt. 
Afterwards, to run a simulation from the container, run the command 

`singularity run simulator_container.sif 10 dstorm` , where 10 is the number of desired simulated samples and dstorm is the microscopy technique (options: dstorm or palm).

If another folder is to be used for models, the container can be run in interactive mode: 

`singularity shell simulator_container.sif`

Edit the config file accordingly and run `python3 /smlm/scripts/run_simulation.py --config new_config.yaml` inside the container with default 15 samples and dstorm technique or `smlm/scripts/run_simulation.py --config new_config.yaml -s number_of_samples -t technique`

#### Contents

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
│   │   pcn.py
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