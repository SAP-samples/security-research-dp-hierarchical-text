import json
import os

import tensorflow as tf

from dph.core.modeling.custom_models import TFBertForHierarchicalClassification
from dph.core.parameters.parameters import Parameters
from dph.core.preprocessing.preprocessor import Preprocessor
from dph.utils.saver import Saver


class ModelLoader:

    def __init__(self, model_path: str, saver: Saver):
        self.model_path = model_path
        self.saver = saver

        with open(self.get_path('parameters.json')) as parameters_json:
            self.parameters = Parameters()
            self.parameters.from_dict(json.load(parameters_json))

    def load_models(self, stacking=1):
        tf.print(f'Loading Models from {self.model_path}...')

        model_folders = [f.name for f in os.scandir(self.get_path('target_model')) if f.is_dir()]
        model_folders.sort()

        loaded_models = []

        if len(model_folders) == 0:  # For compatibility
            loaded_models = [TFBertForHierarchicalClassification.from_pretrained(self.get_path('target_model'))]

        else:
            if len(model_folders) > stacking:
                model_folders = model_folders[-stacking:]
            for model_folder in model_folders:
                model_path = self.get_path('target_model/' + model_folder)
                if hasattr(self.parameters, 'model_path'):
                    if 'bert' in self.parameters.model_path:
                        loaded_models.append(TFBertForHierarchicalClassification.from_pretrained(model_path))
                    else:
                        loaded_models.append(tf.keras.models.load_model(model_path))

        tf.print('loaded models:', len(loaded_models))

        return loaded_models

    def preprocess(self):
        # Compatibility with old saved target models
        if hasattr(self.parameters, "data_set"):
            if self.parameters.data_set == 'BestbuyCleaned':
                self.parameters.dataset = "bestbuy_cleaned"
            elif self.parameters.data_set == 'BestbuyCleaned6040':
                self.parameters.dataset = "bestbuy_cleaned_60-40"
            elif self.parameters.data_set == 'BestbuyCleanedThinned':
                self.parameters.dataset = "bestbuy_cleaned_thinned"
            elif self.parameters.data_set == 'BestbuyCleanedThinned2':
                self.parameters.dataset = "bestbuy_cleaned_thinned_2"

        return Preprocessor().preprocess(self.parameters, self.saver)

    def get_path(self, file_name):
        return os.path.join(self.model_path, file_name)
