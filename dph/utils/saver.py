import json
import os
from datetime import datetime
from pathlib import Path
from random import randint

from dph.core.parameters.parameters import Parameters, DPParameters


class Saver:
    def __init__(self, experiment_name: str, append_date: bool = True, append_random: bool = True):
        self.log_dir = './logs/' + experiment_name \
                       + (('_' + datetime.now().strftime("%Y%m%d-%H%M%S")) if append_date else "") \
                       + (("_" + str(randint(100, 999))) if append_random else "")

        os.makedirs(self.log_dir)

    def save_string(self, string: str, name):
        """
        :param string: string to persist
        :param name: how to name the file
        """
        file = os.path.join(self.log_dir, name) + ".txt"
        with open(file, 'w') as pointer:
            pointer.write(str(string))

    def save_dict(self, dictionary: dict, name):
        """
        :param dictionary: dictionary to persist
        :param name: how to name the file
        """
        file = os.path.join(self.log_dir, name) + ".json"
        with open(file, 'w') as pointer:
            json.dump(dictionary, pointer, indent=2)

    def get_subdirectory(self, name, exist_ok=False):
        """
        :param name: how to name the subdirectory
        :param exist_ok: if false, throws an error if the subdirectory exists
        """
        path = Path(self.log_dir, name)
        path.mkdir(parents=True, exist_ok=exist_ok)
        return str(path)

    @staticmethod
    def get_experiment_name_for_folder_structure(p: Parameters, target: bool):
        dataset_folder = p.dataset
        # noinspection PyUnresolvedReferences
        model_folder = p.model_path
        dp_folder = str(p.noise_multiplier if isinstance(p, DPParameters) else "nodp")
        mia_folder = "target" if target else "attack"
        return str(Path(dataset_folder, model_folder, dp_folder, mia_folder))


class DummySaver(Saver):
    def __init__(self):
        pass

    def save_string(self, string: str, name):
        pass

    def save_dict(self, dictionary: dict, name):
        pass
