from abc import ABC

from tensorflow.python.keras.layers import *
from transformers import add_start_docstrings
from transformers.file_utils import add_start_docstrings_to_callable
from transformers.modeling_tf_bert import BERT_START_DOCSTRING, TFBertPreTrainedModel, BERT_INPUTS_DOCSTRING, \
    TFBertMainLayer
from transformers.modeling_tf_roberta import ROBERTA_START_DOCSTRING, TFRobertaPreTrainedModel, TFRobertaMainLayer, \
    ROBERTA_INPUTS_DOCSTRING
from transformers.modeling_tf_utils import get_initializer
import tensorflow as tf


# noinspection PyAbstractClass
@add_start_docstrings(
    """Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    BERT_START_DOCSTRING,
)
class TFBertForHierarchicalClassification(TFBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.bert = TFBertMainLayer(config, name="bert")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifiers = dict([(level, self.get_dense(level, num_classes, config))
                                 for level, num_classes in config.num_level_classes])

    @staticmethod
    def get_dense(level, num_classes, config):
        return tf.keras.layers.Dense(
            num_classes, kernel_initializer=get_initializer(config.initializer_range), name=level
        )

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def call(self, inputs, output_attack_features=False, **kwargs):
        bert_outputs = self.bert(inputs, **kwargs)

        pooled_output = bert_outputs[1]

        pooled_output = self.dropout(pooled_output, training=kwargs.get("training", False))

        outputs = (dict([(level, classifier(pooled_output)) for level, classifier in self.classifiers.items()]),)

        if output_attack_features:
            outputs += bert_outputs  # add hidden states and attention if they are here

        return outputs  # [(level_name, logits) per level], sequence_output, pooled_output (hidden_states), (attentions)


# noinspection PyAbstractClass
@add_start_docstrings(
    """RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer
    on top of the pooled output) e.g. for GLUE tasks. """,
    ROBERTA_START_DOCSTRING,
)
class TFRobertaForHierarchicalClassification(TFRobertaPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.roberta = TFRobertaMainLayer(config, name="roberta")

        self.classifiers = dict([(level, TFRobertaClassificationHead(num_classes, config, name="classifier"))
                                 for level, num_classes in config.num_level_classes])

    @staticmethod
    def get_dense(level, num_classes, config):
        return tf.keras.layers.Dense(
            num_classes, kernel_initializer=get_initializer(config.initializer_range), name=level
        )

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def call(self, inputs, **kwargs):
        outputs = self.roberta(inputs, **kwargs)

        sequence_output = outputs[0]

        outputs = [(level, classifier(sequence_output, training=kwargs.get("training", False)))
                   for level, classifier in self.classifiers.items()]

        return outputs  # dict of level_name, logits


class TFRobertaClassificationHead(tf.keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, num_labels, config, **kwargs):
        super().__init__(config, **kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.out_proj = tf.keras.layers.Dense(
            num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )

    def call(self, features, training=False):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x, training=training)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        x = self.out_proj(x)
        return x


class SimpleModelForHierarchicalClassification(tf.keras.models.Model, ABC):

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'config': self.config
        })
        return config

    @staticmethod
    def get_dense(level, num_classes):
        return tf.keras.layers.Dense(num_classes, name=level)


class WordCnnForHierarchicalClassification(SimpleModelForHierarchicalClassification):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.embedding = Embedding(input_dim=config.vocab_size,
                                   output_dim=config.embedding_size,
                                   input_length=config.text_length,
                                   weights=config.pretrained_embeddings,
                                   trainable=True)

        del config.pretrained_embeddings
        self.config = config

        # Implementation similar to:
        # https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras/blob/master/sentiment_cnn.py
        # Hyper-Parameters from Paper "CNN for Sentence Classification" (Kim2014)
        filter_sizes = [3, 4, 5]
        num_filters = 100

        self.conv_blocks = []
        for filter_size in filter_sizes:
            conv_block = tf.keras.models.Sequential()
            conv_block.add(Conv1D(num_filters, filter_size, activation='relu'))
            conv_block.add(GlobalMaxPooling1D())
            conv_block.add(Flatten())
            self.conv_blocks.append(conv_block)

        self.concatenate = tf.keras.layers.Concatenate()
        self.dropout = Dropout(0 if hasattr(config, "hidden_dropout_prob") else 0.5)

        self.classifiers = dict([(level, self.get_dense(level, num_classes))
                                 for level, num_classes in config.num_level_classes])

    def call(self, inputs, **kwargs):
        inputs = inputs['input_ids']

        embedded_input = self.embedding(inputs)

        pooled_convs = []
        for conv_block in self.conv_blocks:
            pooled_convs.append(conv_block(embedded_input))

        pooled_convs_concat = self.concatenate(pooled_convs)
        pooled_convs_concat = self.dropout(pooled_convs_concat)

        outputs = (dict([(level, classifier(pooled_convs_concat)) for level, classifier in self.classifiers.items()]),)

        outputs += ([], [pooled_convs_concat], [], [])  # add hidden states and attention if they are here

        return outputs  # [(level_name, logits) per level], [], backbone_output, [], []


class BasicHierarchicalTextClassifier(SimpleModelForHierarchicalClassification):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.backbone = tf.keras.models.Sequential()
        self.backbone.add(Embedding(input_dim=config.vocab_size,
                                    output_dim=config.embedding_size,
                                    input_length=config.text_length,
                                    weights=config.pretrained_embeddings,
                                    trainable=True))

        del config.pretrained_embeddings
        self.config = config

        # self.backbone.add(Dropout(0.2))
        self.backbone.add(GlobalAveragePooling1D())
        # self.backbone.add(Dropout(0.2))

        self.classifiers = dict([(level, self.get_dense(level, num_classes))
                                 for level, num_classes in config.num_level_classes])

    def call(self, inputs, **kwargs):
        inputs = inputs['input_ids']

        backbone_output = self.backbone(inputs)

        outputs = (dict([(level, classifier(backbone_output)) for level, classifier in self.classifiers.items()]),)

        outputs += ([], [backbone_output], [], [])  # add hidden states and attention if they are here

        return outputs  # [(level_name, logits) per level], [], backbone_output, [], []
