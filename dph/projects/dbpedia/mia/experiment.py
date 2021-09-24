from pathlib import Path

from dph.core.parameters.parameters import DPParameters
from dph.mia.attack_parameters import AttackParameters
from dph.mia.mia_experiment import AttackExperiment
from dph.projects.dbpedia.experiments import *
from dph.utils.saver import Saver


def train_ratio_target_models():
    for level_experiment in [DbPediaExperimentL1, DbPediaExperimentL12, DbPediaExperimentL123]:
        for reduced_train_size in int(60e3), int(120e3), int(180e3), None:
            # Target Model
            p = bert_parameters(0)
            p.reduced_train_size = reduced_train_size
            experiment = level_experiment(p)

            # noinspection PyUnresolvedReferences
            experiment_path = Path("dbpedia_ratio",
                                   p.model_path,
                                   str(reduced_train_size),
                                   str(len(experiment.p.levels)))
            experiment.p.experiment_name = str(Path(experiment_path, "target"))

            # Attack Model
            attack_parameters = AttackParameters()
            attack_parameters.experiment_name = str(Path(experiment_path, "attack"))
            attack_experiment = AttackExperiment(attack_parameters, experiment)
            attack_experiment.start()


def train_and_attack_target_model(seed=0):
    for noise_multiplier in [0]:
        # Target Model
        p = bert_parameters(noise_multiplier)
        p.seed += seed
        experiment = DbPediaExperimentL123(p)
        experiment.p.experiment_name = Saver.get_experiment_name_for_folder_structure(p, True)

        attack_parameters = AttackParameters()
        attack_parameters.experiment_name = Saver.get_experiment_name_for_folder_structure(p, False)
        attack_parameters.target_model_path = experiment.saver.log_dir
        attack_parameters.seed += seed
        attack_experiment = AttackExperiment(attack_parameters)
        attack_experiment.start()


def bow_parameters(noise_multiplier):
    if noise_multiplier == 0:
        return BasicClfPretrained.configure(Parameters())
    p = DPParameters(noise_multiplier, clipnorm=0.033400203999999996)
    p = BasicClfPretrained.configure(p)
    p.learning_rate *= 16
    return p


def cnn_parameters(noise_multiplier):
    if noise_multiplier == 0:
        return WordCnnPretrained.configure(Parameters())
    p = DPParameters(noise_multiplier, clipnorm=0.20585039300000002)
    p = WordCnnPretrained.configure(p)
    p.learning_rate *= 1
    return p


def bert_parameters(noise_multiplier):
    if noise_multiplier == 0:
        return BertBase.configure(Parameters())
    p = DPParameters(noise_multiplier, clipnorm=1.59993184, microbatch_size=4)
    p.batch_size_train /= 2
    p = BertBase.configure(p)
    p.learning_rate *= 2
    return p


def stabilize():
    for i in range(3):
        train_and_attack_target_model(i)


if __name__ == '__main__':
    stabilize()
