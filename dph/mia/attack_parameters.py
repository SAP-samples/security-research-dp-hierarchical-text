class AttackParameters(object):
    def __init__(self):
        self.experiment_name = "attack"
        self.seed = 42

        # Path to target model (str)
        # If None, train new target model
        self.target_model_path = None
        self.target_model_stacking = 1

        self.deactivated_components = ['output_attention']
        self.gradient_pca = False

        self.batch_size_train = 32
        self.batch_size_val = 32

        self.num_epochs = 60
        self.patience = 5

        self.model_size = 1
        self.weight_initializer = 'xavier'
        self.activation = 'leaky_relu'
        self.conv_size = 4  # only relevant for active gradients

        self.optimizer = "adam"
        self.learning_rate = 1e-4

        self.dropout_rate = 0.2
        self.batch_normalization = False

        self.log_gradients = False
        self.visualize_loss = False
        self.train_attacker = True
        self.ignore_validation = False

    def from_dict(self, parameters):
        if parameters is not None:
            for parameter_name, value in parameters.items():
                self.__setattr__(parameter_name, value)

    def get_parameters(self):
        return self.__dict__
