import json
from pathlib import Path

import tensorflow as tf
from tensorflow.python.data import Dataset
from tqdm import tqdm


class FeatureSaver:

    def __init__(self, features_path):
        self.features_path = features_path

    def save_features(self, in_datasets, out_datasets):
        self.save_datasets(in_datasets, "in")
        self.save_datasets(out_datasets, "out")

    def save_datasets(self, datasets, folder_name):
        datasets_path = Path(self.features_path, folder_name)
        train, val, test = datasets
        save_dataset(Path(datasets_path, "train"), train)
        save_dataset(Path(datasets_path, "val"), val)
        save_dataset(Path(datasets_path, "test"), test)


def save_dataset(path: Path, dataset: Dataset):
    """
    extract and serialize the Features for the given instances
    """
    assert not path.exists(), "these features seem to already have been written before"
    path.mkdir(parents=True)

    examples_per_file = 5000

    attack_example_writer = None
    attack_features = None
    instance_num = 0

    for instance_num, instance in tqdm(enumerate(dataset)):
        if instance_num % examples_per_file == 0:
            if attack_example_writer is not None:
                attack_example_writer.close()
            attack_example_writer = tf.io.TFRecordWriter(str(Path(path, str(instance_num) + '.tfrecords')),
                                                         tf.io.TFRecordOptions(compression_type='GZIP'))

        attack_features, member = instance

        for key, value in attack_features.items():
            # Saving params
            attack_features[key] = array_feature(value)

        attack_example = tf.train.Example(
            features=tf.train.Features(feature=attack_features)).SerializeToString()
        attack_example_writer.write(attack_example)

    write_dataset_metadata(path, size=instance_num + 1, attack_features=attack_features)


def array_feature(array):
    value = tf.io.serialize_tensor(array)
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_dataset_metadata(file_path: Path, size, attack_features):
    feature_names = list(attack_features.keys())
    metadata = {
        "size": size,
        "feature_names": feature_names
    }

    with open(Path(file_path, 'dataset_metadata.json'), 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)
