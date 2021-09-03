import tensorflow as tf
from anytree import AnyNode
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import pandas as pd

from dph.core.parameters.parameters import Parameters
from dph.core.preprocessing.data import Dataset
from dph.core.preprocessing.dataset_factory import DatasetFactory
from dph.core.preprocessing.one_hot_encoder import OneHotEncoder
from dph.core.preprocessing.text_preprocessor import TextPreprocessorAdvanced, TextPreprocessorSimple
from dph.utils.saver import Saver


class Preprocessor:

    @staticmethod
    def preprocess(p: Parameters, saver: Saver):
        if hasattr(p, "model_path") and hasattr(p, "levels"):
            dataset = Dataset(p.dataset)

            # Reduce training size
            reduced_train_size = None
            if hasattr(p, 'reduced_train_size'):
                if p.reduced_train_size is not None:
                    dataset.train, _removed = train_test_split(dataset.train,
                                                               train_size=p.reduced_train_size,
                                                               random_state=p.seed)
                    reduced_train_size = p.reduced_train_size

            # Preprocess Text
            if hasattr(p, 'embeddings_id'):
                text_preprocessor = TextPreprocessorSimple(dataset, p.features, p.embeddings_id, reduced_train_size)
                p.pretrained_embeddings = text_preprocessor.pretrained_embeddings
            else:
                text_preprocessor = TextPreprocessorAdvanced(dataset, p.features, p.model_path, reduced_train_size)
            train, val, test = text_preprocessor.preprocess()
            train_ids, train_masks, p.train_size = train
            val_ids, val_masks, p.val_size = val
            test_ids, test_masks, p.test_size = test
            p.text_length = len(train_ids[0])

            # Prepare Labels
            p.levels = p.levels if isinstance(p.levels, list) else [p.levels]
            encoder = OneHotEncoder(dataset.train, p.levels)
            labels = encoder.get_labels()
            saver.save_string(str(labels), "class_names")
            train_labels, val_labels, test_labels = encoder.one_hot_all([dataset.train, dataset.val, dataset.test])
            p.num_classes = encoder.get_number_of_classes()
            tf.print("num_classes", p.num_classes)

            assert p.batch_size_train % p.num_gpu == 0, "batch size is not a multiple of num_gpu."

            # Dataset Creation
            dataset_factory = DatasetFactory()
            train_dataset = dataset_factory.create_dataset(train_ids, train_masks, train_labels)
            val_dataset = dataset_factory.create_dataset(val_ids, val_masks, val_labels)
            test_dataset = dataset_factory.create_dataset(test_ids, test_masks, test_labels)

            return labels, train_dataset, val_dataset, test_dataset

    @staticmethod
    def create_hierarchy(p):
        dataset = Dataset(p.dataset)

        # Reduce training size
        if hasattr(p, 'reduced_train_size'):
            if p.reduced_train_size is not None:
                dataset.train, _removed = train_test_split(dataset.train,
                                                           train_size=p.reduced_train_size,
                                                           random_state=p.seed)
        # Hierarchy Creation
        hierarchy = AnyNode(id="root")
        # dataset_concat = pd.concat([dataset.train, dataset.val, dataset.test], ignore_index=True)
        for index, product in tqdm(dataset.train.iterrows(), total=dataset.train.shape[0]):
            parent = hierarchy
            for level in p.levels:
                if pd.notna(product[level]):
                    children_dict = dict([(child.id, child) for child in parent.children])
                    if product[level] not in children_dict:
                        node = AnyNode(id=product[level])
                        node.parent = parent
                    else:
                        node = children_dict[product[level]]
                    parent = node
        return hierarchy
