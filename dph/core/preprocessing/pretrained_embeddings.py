from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm


class PretrainedEmbeddings:

    def __init__(self, folder_path):
        self.embeddings = None
        self.file_path = Path(folder_path, 'embeddings.npy')
        self.vocab_size = None

    def from_folder(self):
        embeddings = np.load(str(self.file_path))
        if len(embeddings) == 1:
            self.vocab_size = int(embeddings[0])
        else:
            self.vocab_size = embeddings.shape[0]
            self.embeddings = [embeddings]

    def to_folder(self, word_index, embeddings_id):
        # Get path to pre-trained embeddings
        embeddings_path = Path('./pretrained_models/embeddings', embeddings_id + '.txt')

        # Create folder for lookup table
        self.file_path.parent.mkdir(parents=True)

        if embeddings_path.exists():
            f = open(embeddings_path)

            tf.print(f"loading embeddings with id {embeddings_id}")
            pretrained_embeddings = {}
            embedding = None
            for line in tqdm(f):
                values = line.split()
                word = values[0]
                embedding = np.asarray(values[1:], dtype='float32')
                pretrained_embeddings[word] = embedding
            f.close()

            assert isinstance(embedding, np.ndarray)

            embeddings = np.zeros((len(word_index) + 1, embedding.shape[0]))
            for word, i in word_index.items():
                embedding_vector = pretrained_embeddings.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embeddings[i] = embedding_vector

            np.save(self.file_path, embeddings)
            self.embeddings = [embeddings]
            self.vocab_size = embeddings.shape[0]
        else:
            self.vocab_size = len(word_index)
            np.save(self.file_path, np.array([len(word_index)]))

    @property
    def embedding_size(self):
        if self.embeddings is not None:
            return self.embeddings[0].shape[1]
        else:
            return 300
