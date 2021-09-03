from pathlib import Path

import tensorflow as tf

from dph.mia.attack_parameters import AttackParameters
from dph.mia.features.extract.model_loader import ModelLoader
from dph.mia.features.extract.feature_extractor import FeatureExtractor
from dph.mia.features.serialize.feature_loader import FeatureLoader
from dph.mia.features.serialize.feature_saver import FeatureSaver
from dph.mia.features.analyze.pca_mapper import PCAMapper
from dph.utils.saver import Saver


class Features:

    def __init__(self, p: AttackParameters, saver: Saver):

        if p.target_model_path.startswith('.'):
            self.model_path = p.target_model_path
        else:
            self.model_path = Path('/mnt/efs/DPTextHierarchy/', p.target_model_path)

        self.saver = saver
        self.p = p
        self.dummy_features = None

    def prepare_datasets(self):

        if hasattr(self.p, "only_use_first_output", ) and self.p.only_use_first_output:
            features_path = Path(self.model_path, "features", "only_first")
            self.p.deactivated_components += ["only_use_first_output"]
        elif hasattr(self.p, "only_use_first_2_outputs", ) and self.p.only_use_first_2_outputs:
            features_path = Path(self.model_path, "features", "only_first_and_second")
            self.p.deactivated_components += ["only_use_first_2_outputs"]
        else:
            features_path = Path(self.model_path, "features", "original")

        if not features_path.exists():
            model_loader = ModelLoader(str(self.model_path), self.saver)
            models = model_loader.load_models(self.p.target_model_stacking)
            labels, train_dataset, val_dataset, test_dataset = model_loader.preprocess()

            if self.p.ignore_validation:
                val_dataset = val_dataset.take(0)

            # Create Feature Extractor
            feature_extractor = FeatureExtractor(models, train_dataset, val_dataset, test_dataset,
                                                 deactivated_components=self.p.deactivated_components)

            in_datasets, out_datasets, min_size = feature_extractor.prepare_datasets()

            feature_saver = FeatureSaver(features_path)
            feature_saver.save_features(in_datasets, out_datasets)

        feature_loader = FeatureLoader(features_path, deactivated_components=self.p.deactivated_components)

        if self.p.gradient_pca:
            features_path = Path(self.model_path, "features", "pca")
            if not features_path.exists():
                in_datasets, out_datasets = feature_loader.load_features()
                train_dataset, val_dataset, test_dataset = create_datasets(in_datasets, out_datasets)

                pca_mapper = PCAMapper()
                pca_mapper.fit(train_dataset)
                in_datasets = pca_mapper.map_transform(in_datasets)
                out_datasets = pca_mapper.map_transform(out_datasets)

                feature_saver = FeatureSaver(features_path)
                feature_saver.save_features(in_datasets, out_datasets)

        feature_loader = FeatureLoader(features_path, deactivated_components=self.p.deactivated_components)
        in_datasets, out_datasets = feature_loader.load_features()
        train_dataset, val_dataset, test_dataset = create_datasets(in_datasets, out_datasets)
        self.dummy_features = feature_loader.dummy_features

        return train_dataset, val_dataset, test_dataset


def create_datasets(in_datasets, out_datasets):
    in_train, in_val, in_test = in_datasets
    out_train, out_val, out_test = out_datasets

    # Merge splits and map each example to features
    train_dataset = tf.data.experimental \
        .choose_from_datasets([in_train.shuffle(500, seed=42),
                               out_train.shuffle(500, seed=42)],
                              tf.data.Dataset.range(2).repeat(in_train.size + out_train.size))
    val_dataset = tf.data.experimental \
        .choose_from_datasets([in_val, out_val], tf.data.Dataset.range(2).repeat(in_val.size + out_val.size))
    test_dataset = tf.data.experimental \
        .choose_from_datasets([in_test, out_test], tf.data.Dataset.range(2).repeat(in_test.size + out_test.size))

    return train_dataset, val_dataset, test_dataset
