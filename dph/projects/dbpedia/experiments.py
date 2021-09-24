from dph.core.experiment import Experiment
from dph.core.parameters.pretrained_models import *


class DbPediaExperiment(Experiment):
    def __init__(self, parameters: Parameters):
        parameters.dataset = "DBPEDIA"
        parameters.features = ["text"]
        parameters.experiment_name += "_dbpedia"
        super(DbPediaExperiment, self).__init__(parameters)


class DbPediaExperimentL1(DbPediaExperiment):
    def __init__(self, parameters: Parameters):
        parameters.levels = ["l1"]
        super().__init__(parameters)


class DbPediaExperimentL12(DbPediaExperiment):
    def __init__(self, parameters: Parameters):
        parameters.levels = ["l1", "l2"]
        super().__init__(parameters)


class DbPediaExperimentL123(DbPediaExperiment):
    def __init__(self, parameters: Parameters):
        parameters.levels = ["l1", "l2", "l3"]
        super().__init__(parameters)


def evaluate():
    p = Parameters()
    p = BertBase.configure(p)
    DbPediaExperimentL123(p).start()


if __name__ == '__main__':
    evaluate()
