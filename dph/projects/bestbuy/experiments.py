from dph.core.experiment import Experiment
from dph.core.parameters.pretrained_models import *


class BestbuyExperiment(Experiment):
    def __init__(self, parameters: Parameters):
        parameters.dataset = "bestbuy_cleaned"
        parameters.features = ["name", "manufacturer", "description"]
        parameters.experiment_name += "_bestbuy"
        super(BestbuyExperiment, self).__init__(parameters)


class BestbuyF1(BestbuyExperiment):
    def __init__(self, parameters: Parameters):
        parameters.levels = "F1"
        parameters.experiment_name += "_F1"
        super(BestbuyF1, self).__init__(parameters)


class BestbuyF2(BestbuyExperiment):
    def __init__(self, parameters: Parameters):
        parameters.levels = "F2"
        parameters.experiment_name += "_F2"
        super(BestbuyF2, self).__init__(parameters)


class BestbuyF3(BestbuyExperiment):
    def __init__(self, parameters: Parameters):
        parameters.levels = "F3"
        parameters.experiment_name += "_F3"
        super(BestbuyF3, self).__init__(parameters)


class BestbuyF123(BestbuyExperiment):
    def __init__(self, parameters: Parameters):
        parameters.levels = ["F1", "F2", "F3"]
        super(BestbuyF123, self).__init__(parameters)


class BestbuyF1234567(BestbuyExperiment):
    def __init__(self, parameters: Parameters):
        parameters.levels = ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]
        super().__init__(parameters)


def evaluate():
    p = Parameters()
    p = WordCnnPretrained.configure(p)
    BestbuyF123(p).start()


if __name__ == '__main__':
    evaluate()
