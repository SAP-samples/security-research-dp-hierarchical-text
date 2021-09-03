from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from abc import ABC, abstractmethod

from dph.core.preprocessing.data import Dataset
from dph.core.preprocessing.pretrained_embeddings import PretrainedEmbeddings


class TextPreprocessor(ABC):

    def __init__(self,
                 dataset: Dataset,
                 feature_names: list,
                 model_name: str,
                 reduced_train_size: int):
        """Text Preprocessor with cache"""
        self.tokenizer = None
        self.dataset = dataset
        self.feature_names = feature_names

        self.id = "_".join([dataset.data_folder, model_name] + feature_names)
        if reduced_train_size is not None:
            self.id += f"_{reduced_train_size}"
        self.folder_path = Path("./data", self.id)

        self.max_len = None

    def preprocess(self):
        return self._preprocess("train"), self._preprocess("val"), self._preprocess("test")

    def _preprocess(self, split_name):
        file_path = Path(self.folder_path, split_name + ".npy")

        if not self.folder_path.exists():
            self.folder_path.mkdir(parents=True)

        if file_path.exists():
            tf.print("Loading tokenized texts")
            input_ids = np.load(str(file_path))
        else:
            texts = self.get_texts(split_name)
            tf.print("Obtaining max_len...")
            self.get_max_len(texts)
            tf.print(f"Tokenizing '{split_name}' with max_len {self.max_len}")
            input_ids = self._tokenize_sentences(texts)
            np.save(file_path, input_ids)

        attention_masks = self._create_attention_masks(input_ids)

        return input_ids, attention_masks, len(input_ids)

    def get_texts(self, split_name):
        dataframe = self.dataset.__getattribute__(split_name)
        texts = dataframe[self.feature_names].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        return texts

    def get_max_len(self, texts):
        if self.max_len:
            return
        self.max_len = int(texts.str.split().apply(len).max())
        tf.print(f"max_len of texts is {self.max_len}")
        tokenizer_max_len = self.tokenizer.max_len if isinstance(self.tokenizer, PreTrainedTokenizer) else 512
        if tokenizer_max_len:
            if tokenizer_max_len < self.max_len:
                tf.print("Caution! Texts are being truncated!")
                self.max_len = tokenizer_max_len

    @abstractmethod
    def _create_attention_masks(self, input_ids):
        pass

    @abstractmethod
    def _tokenize_sentences(self, texts):
        pass


class TextPreprocessorAdvanced(TextPreprocessor):

    def __init__(self, dataset: Dataset, feature_names: list, model_name: str, reduced_train_size: int):
        super().__init__(dataset, feature_names, model_name, reduced_train_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _tokenize_sentences(self, texts):
        input_ids = []

        for sentence in tqdm(texts):
            ids = self.tokenizer.encode(
                sentence,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=self.max_len,
            )

            input_ids.append(ids)

        input_ids = pad_sequences(input_ids,
                                  maxlen=self.max_len,
                                  truncating="post",
                                  padding="post")  # pads to longest sequence

        return input_ids

    def _create_attention_masks(self, input_ids):
        attention_masks = []

        for sentence in input_ids:
            att_mask = [int(token_id > 0) for token_id in sentence]
            attention_masks.append(att_mask)

        return np.asarray(attention_masks)


class TextPreprocessorSimple(TextPreprocessor):

    def __init__(self, dataset: Dataset, feature_names: list, embeddings_id: str, reduced_train_size: int):
        if embeddings_id is None:
            embeddings_id = 'no-embeds'
        super().__init__(dataset, feature_names, embeddings_id, reduced_train_size)

        self.pretrained_embeddings = PretrainedEmbeddings(self.folder_path)
        if self.folder_path.exists():
            self.pretrained_embeddings.from_folder()
        else:
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
            texts = self.get_texts("train")
            self.tokenizer.fit_on_texts(texts)
            self.pretrained_embeddings.to_folder(self.tokenizer.word_index, embeddings_id)

    def _tokenize_sentences(self, texts):
        input_ids = self.tokenizer.texts_to_sequences(texts)

        input_ids = pad_sequences(input_ids,
                                  maxlen=self.max_len,
                                  truncating="post",
                                  padding="post")  # pads to longest sequence

        return input_ids

    def _create_attention_masks(self, input_ids):
        return None
