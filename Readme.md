### PhD Project _"Deep Learning for Pointclouds and particle averaging towards improved single molecule localization microscopy"_

This repository contains scripts, notebooks and helper functions for the PhD project Deep Learning for Pointclouds and particle averaging towards improved single molecule localization microscopy".  

#### Installation 
Using anaconda package manager, one can create a new environment for this repository and install the dependencies by running  

`conda create --name <env_name>`

`python setup.py install`


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

To use the simulation code, one needs to build the original java code of the SureSim simulator from [Kuner Lab](https://github.com/tkunerlab/JavaUmsetzungSTORMSimulation). 
