from abc import ABC

import tensorflow as tf

from dph.mia.attack_parameters import AttackParameters


class AttackModel(tf.keras.Model, ABC):

    def __init__(self, dummy_instance_features: dict, p: AttackParameters):
        super().__init__()

        self.p = p

        # # Parameter handling
        # Initializer
        if self.p.weight_initializer == 'xavier':
            self.p.weight_initializer = tf.keras.initializers.GlorotNormal(seed=42)
        else:
            raise NotImplementedError('This initializer is not implemented:', self.p.activation)

        # Activation function
        if self.p.activation == 'relu':
            self.p.activation = 'relu'
        elif self.p.activation == 'leaky_relu':
            self.p.activation = tf.keras.layers.LeakyReLU(alpha=0.2)
        elif self.p.activation == 'ELU':
            self.p.activation = tf.keras.layers.ELU()
        else:
            raise NotImplementedError('This activation function is not implemented:', self.p.activation)

        # Optimizer
        if self.p.optimizer == 'adam':
            self.p.optimizer = tf.keras.optimizers.Adam(self.p.learning_rate)
        elif self.p.optimizer == 'sgd':
            self.p.optimizer = tf.keras.optimizers.SGD(self.p.learning_rate)

        # Build attack model from dummy instance features
        self.components = dict()

        for feature, dummy_value in dummy_instance_features.items():
            tf.print(feature)
            tf.print(dummy_value.shape)
            if feature.startswith('gradient'):
                if self.p.gradient_pca:
                    self.components[feature] = self.get_fcn_component(dummy_value, name=feature)
                else:
                    self.components[feature] = self.get_gradient_component(dummy_value, name=feature)
                self.components[feature].summary()
            elif feature.startswith('output'):
                self.components[feature] = self.get_fcn_component(dummy_value, name=feature)
            elif feature.startswith('loss'):
                self.components[feature] = self.get_fcn_component(dummy_value, name=feature)
            elif feature.startswith('label'):
                self.components[feature] = self.get_fcn_component(dummy_value, name=feature)
            elif feature.startswith('prediction_confidence'):
                self.components[feature] = self.get_fcn_component(dummy_value, name=feature)
                self.components[feature].summary()
            else:
                raise Exception('Unknown attack feature type!')

        self.concatenate = tf.keras.layers.Concatenate()
        self.encoder_component = self.get_encoder_component()

        self.compile(optimizer=self.p.optimizer,
                     loss='binary_crossentropy',
                     metrics=['binary_accuracy'])

    def call(self, inputs, **kwargs):
        merged_output = []

        for feature, component in sorted(self.components.items()):
            merged_output.append(component((inputs[feature]), **kwargs))

        if len(merged_output) > 1:
            encoder_input = self.concatenate(merged_output)
        else:
            encoder_input = merged_output[0]

        # return merged_output
        return self.encoder_component(encoder_input)

    def get_fcn_component(self, dummy_value=None, name=None):
        """
        The components for the features "output", "loss" and "label" are all the same.
        We call this component fcn component.
        """

        component = tf.keras.Sequential(name=name)
        if dummy_value is not None:
            component.add(tf.keras.Input(shape=dummy_value.shape))
        component.add(tf.keras.layers.Flatten())
        component.add(tf.keras.layers.Dense(128 * self.p.model_size,
                                            activation=self.p.activation,
                                            kernel_initializer=self.p.weight_initializer, name='dense1'))
        component.add(tf.keras.layers.Dropout(self.p.dropout_rate))
        component.add(tf.keras.layers.Dense(64 * self.p.model_size,
                                            activation=self.p.activation,
                                            kernel_initializer=self.p.weight_initializer, name="dense2"))

        return component

    def get_gradient_component(self, dummy_stacked_gradients, name=None):
        """
        The component for "gradient" features.
        """
        component = tf.keras.Sequential(name=name)

        component.add(tf.keras.Input(shape=dummy_stacked_gradients.shape))

        target_layer_input_size = dummy_stacked_gradients.shape[1]

        # we set the size of the convolutional kernel to the input size of the fully connected layer
        component.add(tf.keras.layers.Conv2D(filters=1000 * self.p.model_size,
                                             kernel_size=(target_layer_input_size, self.p.conv_size),
                                             kernel_initializer=self.p.weight_initializer,
                                             data_format='channels_first',
                                             name="grad_conv"))

        if self.p.batch_normalization:
            component.add(tf.keras.layers.BatchNormalization())
        component.add(tf.keras.layers.Dropout(self.p.dropout_rate))
        component.add(self.get_fcn_component())  # re-use fcn_component
        component.add(tf.keras.layers.Dropout(self.p.dropout_rate))

        return component

    def get_encoder_component(self):
        """Encoder component at end of attack model"""

        encoder_component = tf.keras.Sequential(name='encoder')

        encoder_component.add(tf.keras.layers.Dense(256 * self.p.model_size,
                                                    activation=self.p.activation,
                                                    kernel_initializer=self.p.weight_initializer))
        encoder_component.add(tf.keras.layers.Dropout(self.p.dropout_rate))
        encoder_component.add(tf.keras.layers.Dense(128,
                                                    activation=self.p.activation,
                                                    kernel_initializer=self.p.weight_initializer))
        encoder_component.add(tf.keras.layers.Dropout(self.p.dropout_rate))
        encoder_component.add(tf.keras.layers.Dense(64,
                                                    activation=self.p.activation,
                                                    kernel_initializer=self.p.weight_initializer))
        encoder_component.add(tf.keras.layers.Dense(1,
                                                    activation="sigmoid", kernel_initializer=self.p.weight_initializer,
                                                    name="output"))
        return encoder_component
