from collections import OrderedDict

import tensorflow as tf
from transformers import TFPreTrainedModel


class FeatureExtractor:
    """
    Class to extract features for MI Attack
    """

    def __init__(self, models: list,
                 train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset,
                 deactivated_components: list = None):
        """
        :param models: Target model list. Multiple models if features should be stacked.
        :param train_dataset: Train Dataset
        :param val_dataset: Val Dataset
        :param test_dataset: Test Dataset
        :param deactivated_components: Deactivated Components
        """
        self.models = models
        self.instances_in = train_dataset
        self.instances_out = val_dataset.concatenate(test_dataset)

        self.activated_components = {'gradient', 'output_hidden', 'output', 'loss', 'label', 'output_attention'}

        if deactivated_components is not None:
            self.activated_components -= set(deactivated_components)
            tf.print("deactivated_components", deactivated_components)
            tf.print("activated_components", self.activated_components)

        tf.print("Instantiated feature extractor with activated components:\n", self.activated_components)

    def prepare_datasets(self):
        out_all = self.instances_out
        in_all = self.instances_in

        # Find out length of each dataset
        out_size = sum(1 for _ in self.instances_out)
        in_size = sum(1 for _ in self.instances_in)
        min_size = min([out_size, in_size])
        tf.print(f"out_size: {out_size}, in_size: {in_size}, min_size: {min_size}")

        # Make both datasets same length
        in_all = in_all.take(min_size)
        out_all = out_all.take(min_size)

        # Add membership information
        out_all = out_all.map(lambda inputs, label: (inputs, label, 0))
        in_all = in_all.map(lambda inputs, label: (inputs, label, 1))

        # Split
        in_datasets = split_dataset(in_all, min_size)
        out_datasets = split_dataset(out_all, min_size)

        in_datasets = self.map_datasets(in_datasets)
        out_datasets = self.map_datasets(out_datasets)

        return in_datasets, out_datasets, min_size

    def map_datasets(self, datasets):
        train, val, test = datasets
        return train.map(self.get_mapping_function()), \
               val.map(self.get_mapping_function()), \
               test.map(self.get_mapping_function())

    def get_mapping_function(self):

        # Define compute_loss Function for loss feature extraction
        loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        def compute_loss(labels, prediction):
            loss = 0
            for index, level_name in enumerate(sorted(prediction.keys())):
                loss += loss_object(labels[index], prediction[level_name][0])

            return loss

        def map_instance_to_attack_features(inputs, label, member):
            attack_features = self.calculate_attack_features(inputs, label, compute_loss)
            return attack_features, member

        return map_instance_to_attack_features

    @tf.function
    def calculate_attack_features(self, inputs, label, compute_loss):
        inputs = {'input_ids': tf.expand_dims(inputs['input_ids'], 0),
                  'attention_mask': tf.expand_dims(inputs['attention_mask'], 0)}

        attack_features = dict()

        def stack_features(feature_name: str, feature_value):
            if feature_name.startswith('label'):
                # Do not stack labels
                attack_features[feature_name] = feature_value
            else:
                if feature_name in attack_features:
                    attack_features[feature_name] = tf.concat(
                        [attack_features[feature_name], tf.expand_dims(feature_value, 0)], 0)
                else:
                    attack_features[feature_name] = tf.expand_dims(feature_value, 0)

        for model in self.models:

            if 'gradient' in self.activated_components:
                watched_variables = model.trainable_variables[-6:]
                # print("Watched Variables:")
                # for variable in watched_variables:
                #     print(variable.name, variable.shape)

            with tf.GradientTape() as gradient_tape:
                kwargs = {}
                if isinstance(model, TFPreTrainedModel):
                    kwargs['output_attack_features'] = True
                prediction, sequence_output, pooled_output, hidden_states, attentions = \
                    model(inputs, **kwargs)

                loss = compute_loss(label, prediction)

                if 'loss' in self.activated_components:
                    stack_features('loss', [loss])

                if 'gradient' in self.activated_components:
                    with gradient_tape.stop_recording():
                        # noinspection PyUnboundLocalVariable
                        gradients = gradient_tape.gradient(loss, watched_variables)

                    # In the following, we concatenate the gradients of input neurons and bias
                    gradient_features = OrderedDict()
                    for variable, gradient in zip(watched_variables, gradients):
                        if isinstance(gradient, tf.IndexedSlices):
                            gradient = gradient.values  # Maybe we should ignore IndexedSlices?
                        gradient_layer_name = '/'.join(variable.name.split('/')[1:-1])  # find out layer name
                        if gradient_layer_name not in gradient_features:
                            gradient_features[gradient_layer_name] = gradient
                        else:
                            gradient_features[gradient_layer_name] = \
                                tf.concat((gradient_features[gradient_layer_name], [gradient]), 0)

                    # Append concatenated gradients
                    for gradient_name, gradient in gradient_features.items():
                        stack_features('gradient_' + gradient_name, gradient)

            if 'output_attention' in self.activated_components:
                for num, attention in enumerate(attentions):
                    # fist [0] for first instance in batch, second [0] for 'CLS' output
                    cls_attentions = tf.reshape(attention[0, :, 0, :], (12, 101))
                    for head_num in range(12):
                        stack_features('output_attention_' + str(num) + str(head_num),
                                       cls_attentions[head_num])

            if 'output_hidden' in self.activated_components:
                for num, hidden_state in enumerate(hidden_states):
                    # fist [0] for first instance in batch, second [0] for 'CLS' output
                    stack_features('output_hidden_' + str(num), hidden_state[0][0])

            if 'output' in self.activated_components:
                stack_features('output_hidden_pooled', pooled_output[0])

                for level_name, level_output in prediction.items():
                    # [0] for first instance in batch
                    stack_features('output_' + str(level_name), tf.nn.softmax(level_output[0]))

            if 'label' in self.activated_components:
                for num, label_level in enumerate(label):
                    stack_features('label' + str(num), label_level)

        return attack_features


def split_dataset(dataset_all, dataset_size):
    train_size = int(0.8 * dataset_size)
    # val_size = int(0.1 * dataset_size)
    test_size = int(0.1 * dataset_size)

    # train_dataset = dataset_all.take(train_size)
    # test_dataset = dataset_all.skip(train_size)
    # val_dataset = test_dataset.take(val_size)
    # test_dataset = test_dataset.skip(val_size)

    test_dataset = dataset_all.take(test_size)
    train_dataset = dataset_all.skip(test_size)
    val_dataset = train_dataset.skip(train_size)
    train_dataset = train_dataset.take(train_size)

    return train_dataset, val_dataset, test_dataset
