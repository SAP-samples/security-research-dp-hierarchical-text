import json
import os
from pathlib import Path

import tensorflow as tf


class FeatureLoader:

    def __init__(self, features_path, deactivated_components):
        self.features_path = features_path
        self.dummy_features = None
        self.deactivated_components = deactivated_components

    def load_features(self):
        in_datasets = self.load_datasets("in", member=1)
        out_datasets = self.load_datasets("out", member=0)

        return in_datasets, out_datasets

    def load_datasets(self, folder_name, member: int):
        datasets_path = Path(self.features_path, folder_name)
        train = self.load_dataset(Path(datasets_path, "train"), member)
        val = self.load_dataset(Path(datasets_path, "val"), member)
        test = self.load_dataset(Path(datasets_path, "test"), member)

        return train, val, test

    def load_dataset(self, path: Path, member: int):
        with open(Path(path, 'dataset_metadata.json')) as metadata_file:
            metadata = json.load(metadata_file)

        files = [os.path.join(str(path), f) for f in os.listdir(str(path)) if
                 os.path.isfile(os.path.join(str(path), f)) and "metadata" not in f]
        tf.print(f"Using {str(len(files))} files from {str(path)}.")

        dataset = tf.data.TFRecordDataset(files, compression_type='GZIP', num_parallel_reads=len(files))

        # Parse lines
        tf.print('map')
        dataset = dataset.map(self.get_parser_function(metadata['feature_names'], member=member))

        if self.dummy_features is None:
            self.dummy_features = list(dataset.take(1))[0][0]

        dataset = dataset.map(self.get_ensure_shape_function())

        dataset.size = metadata['size']

        return dataset

    def get_parser_function(self, feature_names, member):

        def load_feature(example_message, feature_name):
            b_feature = example_message[feature_name]
            if feature_name.startswith('label'):
                feature_value = tf.io.parse_tensor(b_feature, out_type=tf.int32)
            else:
                feature_value = tf.io.parse_tensor(b_feature, out_type=tf.float32)
            # if self.dummy_features is not None:
            #     feature_value = tf.ensure_shape(feature_value, self.dummy_features[feature_name].shape)
            return feature_value

        def parse_attack_features(element):
            parse_dic = {}
            for feature_name in feature_names:
                parse_dic[feature_name] = tf.io.FixedLenFeature([], tf.string)

            example_message = tf.io.parse_single_example(element, parse_dic)

            features = {}

            for feature_name in feature_names:
                if any([feature_name.startswith(deactivated) for deactivated in self.deactivated_components]):
                    continue

                feature_value = load_feature(example_message, feature_name)

                features[feature_name] = feature_value

            if 'prediction_confidence' not in self.deactivated_components:
                level_output_names = \
                  [feature_name for feature_name in feature_names if feature_name.startswith(('output_F', 'output_l'))]

                if set(level_output_names).issubset(set(features)):
                    level_outputs = [features[level_output_name] for level_output_name in level_output_names]
                else:
                    # We have to load the level outputs
                    level_outputs = \
                        [load_feature(example_message, level_output_name) for level_output_name in level_output_names]

                prediction_confidence = tf.constant(1.0)
                for level_output_name, level_output in zip(level_output_names, level_outputs):
                    prediction_confidence *= tf.math.reduce_max(level_output)

                features['prediction_confidence'] = prediction_confidence

            return features, member

        return parse_attack_features

    def get_ensure_shape_function(self):
        def ensure_shape(features, member):
            for key in features:
                features[key] = tf.ensure_shape(features[key], self.dummy_features[key].shape)

            return features, member

        return ensure_shape
