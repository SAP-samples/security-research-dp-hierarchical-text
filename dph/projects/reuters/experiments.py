from dph.core.experiment import Experiment
from dph.core.parameters.pretrained_models import *


class ReutersExperiment(Experiment):
    def __init__(self, parameters: Parameters):
        parameters.dataset = "reuters"
        parameters.features = ["headline", "text"]
        parameters.experiment_name += "_reuters"
        super().__init__(parameters)


class ReutersL123(ReutersExperiment):
    def __init__(self, parameters: Parameters):
        parameters.levels = ["l1", "l2", "l3"]
        super().__init__(parameters)


class ReutersL12(ReutersExperiment):
    def __init__(self, parameters: Parameters):
        parameters.levels = ["l1", "l2"]
        super().__init__(parameters)


class ReutersL1(ReutersExperiment):
    def __init__(self, parameters: Parameters):
        parameters.levels = ["l1"]
        super().__init__(parameters)


def evaluate():
    p = Parameters()
    p = BertBase.configure(p)
    ReutersL123(p).start()


if __name__ == '__main__':
    evaluate()
