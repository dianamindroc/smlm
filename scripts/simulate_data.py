# Simulate point clouds with SureSim

from simulator.simulate import Simulate
import scripts.config as c

simulator = Simulate(c.jar_path, c.path, 'adfl-r-cut.wimp')
simulator.get_params_dict()

simulator.params_dict['detectionEfficiency'] = 40.0
simulator.params_dict['epitopeDensity'] = 0.325
simulator.params_dict['recordedFrames'] = 5000

simulator.update_simulation_params('less_localizations_cut')

simulator.simulate(output='output')