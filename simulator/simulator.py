"""
Class for simulating SMLM data using SuReSim simulator (http://www.ana.uni-heidelberg.de/?id=198)
For examples of model and simulation parameters files, go to https://github.com/tkunerlab/JavaUmsetzungSTORMSimulation/tree/master/examples/cli_example
"""
import os
import subprocess as sp
import json
import shutil


class Simulator:
    def __init__(self, jar: str, path: str, model: str, sim_params: dict):
        """
        Initialization function
        :param jar: path to the suresim .jar file
        :param path: path to current working directory
        :param sim_params: path to the simulation folder
        """
        self.jar = jar
        self.path = path
        self.model = os.path.join(self.path, model)
        self.params = sim_params
        with open(os.path.join(self.path, 'simulationParameters.json'), 'w') as fp:
            json.dump(self.params, fp)
            fp.close()
        self.params_json = os.path.join(self.path, 'simulationParameters.json')

    def get_params_dict(self) -> dict:
        """
        Gets the parameters file as a dictionary. Can then be modified as desired and reassigned.
        :return: dictionary
        """
        # p = open(self.parameters)
        # self.params_dict = json.load(p)
        return self.params

    # def update_simulation_params(self, folder_name : str):
    #     """
    #     Method to update the simulation parameters for a new simulation and make a new directory with the new file
    #     :param folder_name: name for the new simulation folder
    #     :return:
    #     """
    #     self.new_path = os.path.join(self.path, folder_name)
    #     os.mkdir(self.new_path)
    #     with open(os.path.join(self.new_path, 'simulationParameters.json'), 'w') as fp:
    #         json.dump(self.params, fp)
    #     self.params_json = os.path.join(self.new_path, 'simulationParameters.json')
    #     self.model = shutil.copy(self.model, self.new_path)
    #     self.path = self.new_path

    def simulate(self, output):
        """
        Method which actually calls the .jar file and runs the simulation
        :param output: name of output folder
        :return: None
        """
        cmd = ['java', '-jar', self.jar, self.model, self.params_json, os.path.join(self.path, output)]
        sp.check_output(cmd)
        shutil.copy(self.params_json, os.path.join(self.path, output))

