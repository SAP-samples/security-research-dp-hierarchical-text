from abc import ABC, abstractmethod

from dph.core.parameters.parameters import Parameters


class PretrainedModel(ABC):
    @staticmethod
    @abstractmethod
    def configure(parameters: Parameters):
        raise NotImplementedError


class XLNet(PretrainedModel):
    @staticmethod
    def configure(parameters: Parameters):
        parameters.model_path = 'xlnet-base-cased'
        parameters.learning_rate = 3e-5
        parameters.experiment_name += 'xlnet'
        return parameters


class RobertaBase(PretrainedModel):
    @staticmethod
    def configure(parameters: Parameters):
        parameters.model_path = 'roberta-base'
        parameters.learning_rate = 3e-5
        parameters.experiment_name += 'roberta_base'
        return parameters


class BertBase(PretrainedModel):
    @staticmethod
    def configure(parameters: Parameters):
        parameters.model_path = 'bert-base-uncased'
        parameters.learning_rate = 3e-5
        parameters.experiment_name += 'bert_base'
        return parameters


class WordCnnScratch(PretrainedModel):
    @staticmethod
    def configure(parameters: Parameters):
        parameters.model_path = 'WordCnn'
        parameters.learning_rate = 0.001
        parameters.embeddings_id = None
        parameters.experiment_name += 'word_cnn'
        return parameters


class WordCnnPretrained(PretrainedModel):
    @staticmethod
    def configure(parameters: Parameters):
        WordCnnScratch.configure(parameters)
        parameters.embeddings_id = 'glove.6B.300d'
        parameters.experiment_name += '_pretrained'
        return parameters


class BasicClfScratch(PretrainedModel):
    @staticmethod
    def configure(parameters: Parameters):
        parameters.model_path = 'BasicClf'
        parameters.learning_rate = 0.001
        parameters.embeddings_id = None
        parameters.experiment_name += 'basic'
        return parameters


class BasicClfPretrained(PretrainedModel):
    @staticmethod
    def configure(parameters: Parameters):
        BasicClfScratch.configure(parameters)
        parameters.embeddings_id = 'glove.6B.300d'
        parameters.experiment_name += '_pretrained'
        return parameters


class BertSmall(PretrainedModel):
    @staticmethod
    def configure(parameters: Parameters):
        parameters.model_path = './pretrained_models/bert-small-uncased'  # Small
        parameters.learning_rate = 3e-5
        parameters.experiment_name += 'bert_small'
        return parameters


class BertTiny(PretrainedModel):
    @staticmethod
    def configure(parameters: Parameters):
        parameters.model_path = './pretrained_models/bert-tiny-uncased'  # Tiny
        parameters.learning_rate = 3e-5
        parameters.experiment_name += 'bert_tiny'
        return parameters


class BertTinyReinit(PretrainedModel):
    @staticmethod
    def configure(parameters: Parameters):
        BertTiny.configure(parameters)
        Reinit.configure(parameters)
        return parameters


class Reinit(PretrainedModel):
    @staticmethod
    def configure(parameters: Parameters):
        parameters.random_init = True
        parameters.experiment_name += '_reinit'
        return parameters


class ReinitXavier(PretrainedModel):
    @staticmethod
    def configure(parameters: Parameters):
        parameters.random_init = "xavier"
        parameters.experiment_name += '_xavier_reinit'
        return parameters


class OnlyDense(PretrainedModel):
    @staticmethod
    def configure(parameters: Parameters):
        parameters.only_dense = True
        parameters.num_epochs = 50
        parameters.learning_rate = 0.01
        parameters.experiment_name += "_only_dense"
        return parameters
