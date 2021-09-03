import tensorflow as tf

from dph.core.evaluation.evaluator import Evaluator
from dph.core.parameters.parameters import Parameters
from dph.core.preprocessing.preprocessor import Preprocessor
from dph.core.training.trainer import Trainer
from dph.utils.saver import Saver
from dph.utils.set_seed import set_seed


class Experiment:
    def __init__(self,
                 p: Parameters):
        """
        :param p: Parameters object

        :returns history: A keras History object
        """

        self.saver = None  # Enable changing the experiment folder
        self.p = p

        set_seed(p.seed)

    def start(self):
        self.saver = Saver(self.p.experiment_name)

        labels, train_dataset, val_dataset, test_dataset = \
            Preprocessor().preprocess(self.p, self.saver)

        if self.p.experiment_name.startswith("debug"):
            tf.print('*' * 50, "DEBUGGING MODE", '*' * 50)
            self.p.num_epochs = 1
            train_dataset = train_dataset.take(self.p.batch_size_train)
            val_dataset = val_dataset.take(self.p.batch_size_train)
            test_dataset = test_dataset.take(self.p.batch_size_val)
            tf.config.experimental_run_functions_eagerly(True)

        model, history = Trainer().train(train_dataset, val_dataset, self.p, self.saver)

        hierarchy = Preprocessor().create_hierarchy(self.p)
        Evaluator().evaluate(self.saver, "train", labels, model, train_dataset, hierarchy, self.p)
        Evaluator().evaluate(self.saver, "test", labels, model, test_dataset, hierarchy, self.p)

        return history
