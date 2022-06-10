### PhD Project _"Deep Learning for Pointclouds and particle averaging towards improved single molecule localization microscopy"_

This repository contains scripts, notebooks and helper functions for the PhD project Deep Learning for Pointclouds and particle averaging towards improved single molecule localization microscopy".  

#### Installation 
Using anaconda package manager, one can create a new environment for this repository and install the dependencies by running  

`conda create --name <env_name>`

`pip install .` in the smlm folder 


#### Contents

```
smlm
│   README.md  
│   setup.py
└───helpers
│   │   readers.py
│   │   visualization.py
└───notebooks
│   │   visualization.ipynb
│   │   smlm_preprocessing.ipynb 
└───scripts
│   │   run_simulation.py
│   │   config.yaml
└───simulator
│   │   simulator.py
│   │   simulate_data.py
└──────  
```

#### Simulations

To use the simulation code, the original java code of the SureSim simulator from [Kuner Lab](https://github.com/tkunerlab/JavaUmsetzungSTORMSimulation) needs to be built. 
Subsequently, the path to the .jar file (the target of the project) will be used in the configuration file `config.yaml`. 
To run the simulations from the command line, navigate to the scripts folder and run simulations:

`python run_simulation.py --config config.yaml`

Default number of simulated samples is 15. The default microscopy technique used for simulation parameters is dSTORM. Epitope density, recorded frames and detection efficiency are the simulation parameters that are randomly varied in specific ranges. To modify these ranges, navigate to `simulate_data.py` in `simulation` folder.

#### Containerized application

Prerequisites: [Singularity](https://sylabs.io/guides/3.0/user-guide/quick_start.html) installed on the host machine.

The simulator is containerized in a Singularity container and can be obtained by building the container from the definition file provided in `/container/simulator.def` with the command 

`singularity build simulator_container.sif simulator.def`

or downloading it from Singularity Hub 

`singularity pull library://dianamindroc/smlm_simulator/simulation_container`

The container expects a folder `model` in the working directory which contains the ground truth model to simulate samples from. Accepted formats are .wimp and .txt. 
Afterwards, to run a simulation from the container, run the command 

`singularity run simulator_container.sif 10 dstorm` , where 10 is the number of desired simulated samples and dstorm is the microscopy technique (options: dstorm or palm).

If another folder is to be used for models, the container can be run in interactive mode: 

`singularity shell simulator_container.sif`

Edit the config file accordingly and run `python3 /smlm/scripts/run_simulation.py --config new_config.yaml` with default 15 samples and dstorm technique or `smlm/scripts/run_simulation.py --config new_config.yaml -s number_of_samples -t technique`

