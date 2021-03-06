# Simulate point clouds with SureSim

from simulation.simulator import Simulator
import os
from random import choice
import numpy as np
import shutil

technique = {
    "dstorm": {"angleOfLabel": 1.54,
               "angularDeviation": 0.1,
               "detectionEfficiency": 40.0,
               "labelingEfficiency": 50.0,
               "labelEpitopeDistance": 16.0,
               "fluorophoresPerLabel": 1.0,
               "recordedFrames": 10000,
               "dutyCycle": 0.0005,
               "sigmaXY": 10.0,
               "sigmaZ": 40.0,
               "MeanPhotonNumber": 4000,
               "radiusOfFilament": 12.5,
               "epitopeDensity": 0.0625,
               "coupleSigmaIntensity": 'true',
               "makeItReproducible": 'true',
               "viewStatus": 1,
               "pixelsize": 100.0,
               "sigmaRendering": 100.0,
               "colorProof": 'true',
               "useSTORMBlinking": 'true'
               },
    "palm":  {"angleOfLabel": 1.54,
              "angularDeviation": 0.1,
              "labelingEfficiency": 99.0,
              "labelEpitopeDistance": 16.0,
              "fluorophoresPerLabel": 1.0,
              "recordedFrames": 10000,
              "dutyCycle": 0.0005,
              "sigmaXY": 10.0,
              "sigmaZ": 40.0,
              "MeanPhotonNumber": 4000,
              "radiusOfFilament": 12.5,
              "epitopeDensity": 0.0625,
              "coupleSigmaIntensity": 'true',
              "makeItReproducible": 'true',
              "viewStatus": 1,
              "pixelsize": 100.0,
              "sigmaRendering": 100.0,
              "colorProof": 'true',
              "useSTORMBlinking": 'false'
              }
}


def simulate(config, samples, tech, split):
    detection_efficiency = np.round(np.linspace(40, 80, 20), 2)
    epitope_density = np.round(np.linspace(0.01, 0.2, 100), 2)
    recorded_frames = np.round(np.linspace(1000, 4000, 20))

    for file in os.listdir(config['models']):
        if file.endswith('.wimp'):
            model = file
    for n in range(0, samples):
        parameters = technique[tech]
        parameters['detectionEfficiency'] = choice(detection_efficiency)
        parameters['epitopeDensity'] = choice(epitope_density)
        parameters['recordedFrames'] = choice(recorded_frames)
        simulator = Simulator(config['jar_path'], config['models'], model, parameters)
        name = 'sample_' + str(n)
        simulator.simulate(output=name)
        print('Sample ' + str(n) + ' done.')
    if split:
        train_path = os.path.join(config['models'], 'train')
        test_path = os.path.join(config['models'], 'test')
        if not os.path.exists(train_path):
            os.mkdir(train_path)
        if not os.path.exists(test_path):
            os.mkdir(test_path)
        train = 80/100 * samples
        for n in range(0, int(train)):
            shutil.move(os.path.join(config['models'], 'sample_'+str(n)), train_path)
        for m in range(int(train), samples):
            shutil.move(os.path.join(config['models'], 'sample_'+str(m)), test_path)

