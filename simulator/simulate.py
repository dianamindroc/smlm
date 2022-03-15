"""
Class for simulating SMLM data using SuReSim simulator (http://www.ana.uni-heidelberg.de/?id=198)
For examples of model and simulation parameters files, go to https://github.com/tkunerlab/JavaUmsetzungSTORMSimulation/tree/master/examples/cli_example
"""
import os
import subprocess as sp
import json
import shutil

class Simulate():
    def __init__(self, jar: str, path: str, current_folder: str):
        """
        Initialization function
        :param jar: path to the suresim .jar file
        :param path: path to current working directory
        :param parameters: path to the simulation folder
        """
        self.jar = jar
        self.path = path
        self.folder = current_folder
        self.model = os.path.join(self.path, self.folder, 'model_file.txt')
        self.parameters = os.path.join(self.path, self.folder, 'simulationParameters.json')

    def get_params_dict(self) -> dict:
        """
        Gets the parameters file as a dictionary. Can then be modified as desired and reassigned.
        :return: dictionary
        """
        p = open(self.parameters)
        self.params_dict = json.load(p)
        return self.params_dict

    def update_simulation_params(self, folder_name : str):
        """
        Method to update the simulation parameters for a new simulation and make a new directory with the new file
        :param folder_name: name for the new simulation folder
        :return:
        """
        self.folder = os.path.join(self.path, folder_name)
        os.mkdir(self.folder)
        with open(os.path.join(self.folder, 'simulationParameters.json'), 'w') as fp:
            json.dump(self.params_dict, fp)
        self.parameters = os.path.join(self.folder, 'simulationParameters.json')
        self.model = shutil.copy(self.model, self.folder)

    def simulate(self, output):
        """
        Method which actually calls the .jar file and runs the simulation
        :param output: name of output folder
        :return: None
        """
        self.cmd = ['java', '-jar', self.jar, self.model, self.parameters, os.path.join(self.path, self.folder, output)]
        sp.check_output(self.cmd)

