import os
from pathlib import Path

import tensorflow as tf
from tqdm.auto import tqdm

from dph.mia.features.features import Features


def visualize_loss(features: Features, log_dir):
    Path(log_dir).mkdir(parents=True)

    train_dataset, val_dataset, test_dataset = features.prepare_datasets()

    dataset = train_dataset.concatenate(val_dataset).concatenate(test_dataset)

    for attack_features, member in tqdm(dataset):
        if member == 1:
            tf.print(attack_features['loss'],
                     output_stream='file://' + os.path.join(log_dir, 'in_loss.txt'))
        else:
            assert member == 0
            tf.print(attack_features['loss'],
                     output_stream='file://' + os.path.join(log_dir, 'out_loss.txt'))
