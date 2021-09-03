import tensorflow as tf


class Parameters(object):
    def __init__(self):
        self.experiment_name = ""
        self.dataset = None
        self.batch_size_train = 32
        self.batch_size_val = 32
        self.num_epochs = 30
        self.features = None
        self.seed = 42
        self.learning_rate = None
        self.patience = 3
        self.num_gpu = len(tf.config.list_physical_devices('GPU')) or 1

    def from_dict(self, parameters):
        if parameters is not None:
            for parameter_name, value in parameters.items():
                self.__setattr__(parameter_name, value)

    def get_parameters(self):
        return self.__dict__


class DPParameters(Parameters):
    def __init__(self, noise_multiplier, clipnorm, microbatch_size=1):
        super(DPParameters, self).__init__()
        self.sigma = None
        self.experiment_name = "dp_"
        self.noise_multiplier = noise_multiplier
        self.clipnorm = clipnorm
        # Set num_microbatches_per_gpu to batch_size by default
        self.microbatch_size = microbatch_size
        self.hidden_dropout_prob = 0
        self.batch_size_train *= 2
        self.batch_size_val *= 2
        self.num_epochs = 100


class AdamWParameters(Parameters):
    def __init__(self):
        super(AdamWParameters, self).__init__()
        self.num_epochs = 2
        self.experiment_name = "w_"
