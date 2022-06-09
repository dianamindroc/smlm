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

The simulator is containerized in a Singularity container and can be obtained by pulling the image from Singularity Hub: 

`singularity pull --arch amd64 library://dianamindroc/smlm_simulator/simulator_container:latest`

Prerequisite is installing singularity on the local machine. 
To use the container, run the singularity run command together with the configuration .yaml file (see `/scripts/config.yaml) `that specifies the desired model: 

`singularity run simulator_container.sif config.yaml`