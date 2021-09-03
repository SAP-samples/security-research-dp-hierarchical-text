from sklearn.decomposition import IncrementalPCA
import tensorflow as tf
from tqdm import tqdm


class PCAMapper:

    def __init__(self):
        self.pca_by_feature = {}
        self.num_components_by_feature = {}

    def fit(self, train_dataset):
        # normalization_path_on_efs = Path('/mnt/efs/DPTextHierarchy/logs', 'gradient_pca')

        # if normalization_path_on_efs.exists():
        #     self.pca_by_feature = pickle.load(open(normalization_path_on_efs, "rb"))

        # else:
        if True:  # Delete this line if you want to pickle pcas
            # This code segment only works with environment variable OMP_NUM_THREADS = 1

            n_components = 800
            for attack_features, member in tqdm(train_dataset.batch(n_components, drop_remainder=True)):
                tf.print("Fitting PCA ...")
                for feature, value in attack_features.items():
                    if 'gradient' in feature:
                        if feature not in self.pca_by_feature:
                            self.pca_by_feature[feature] = IncrementalPCA(batch_size=n_components,
                                                                          n_components=n_components)
                        partial_fit_x = tf.reshape(value, tuple(value.shape[:1]) + (-1,))  # Flatten gradient
                        self.pca_by_feature[feature].partial_fit(partial_fit_x)

            # pickle.dump(self.pca_by_feature, open(normalization_path_on_efs, 'wb'))

        for feature, inc_pca in self.pca_by_feature.items():
            explained_variance = 0
            num_components = 0
            for variance in inc_pca.explained_variance_ratio_:
                explained_variance += variance
                num_components += 1
                if explained_variance > 0.95:
                    break
                self.num_components_by_feature[feature] = num_components

        tf.print(self.num_components_by_feature)

    def transform(self, attack_features_, member_):
        def pca_transform_tf(inc_pca_, x: tf.Tensor):
            x = x - inc_pca_.mean_
            x_transformed = tf.keras.backend.dot(x, tf.constant(inc_pca_.components_.T, dtype=x.dtype))
            return x_transformed

        for feature_, value_ in attack_features_.items():
            if 'gradient' in feature_:
                value_ = tf.reshape(value_, tuple(value_.shape[:1]) + (-1,))
                value_ = pca_transform_tf(self.pca_by_feature[feature_], value_)[0]
                attack_features_[feature_] = value_[:self.num_components_by_feature[feature_]]

        return attack_features_, member_

    def map_transform(self, datasets):
        train, val, test = datasets
        return train.map(self.transform), val.map(self.transform), test.map(self.transform)
