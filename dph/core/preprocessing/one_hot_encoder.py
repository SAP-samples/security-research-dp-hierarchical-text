from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
import pandas as pd
import numpy as np


class OneHotEncoder:
    def __init__(self, data_frame: pd.DataFrame, levels):
        self.levels = levels
        self.encoder = SklearnOneHotEncoder(dtype=np.int32, handle_unknown='ignore')
        self.encoder.fit(self._to_dataframe(data_frame[levels]))

    def one_hot(self, data_frame: pd.DataFrame):
        merged = self.encoder.transform(self._to_dataframe(data_frame[self.levels])).toarray()
        number_of_classes = self.get_number_of_classes()
        begin_columns = [sum(number_of_classes[:i]) for i in range(len(number_of_classes))]
        end_columns = [sum(number_of_classes[:i + 1]) for i in range(len(number_of_classes))]
        splitted = [merged[:, begin_columns[i]:end_columns[i]] for i in range(len(self.get_number_of_classes()))]
        return splitted

    def one_hot_all(self, data_frames: list):
        return [self.one_hot(data_frame) for data_frame in data_frames]

    def get_number_of_classes(self):
        return [len(level) for level in self.encoder.categories_]

    def get_labels(self):
        return self.encoder.categories_

    @staticmethod
    def _to_dataframe(columns):
        return pd.DataFrame(columns).fillna("nan")
