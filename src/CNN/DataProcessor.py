import re
import string
from tqdm import tqdm
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.sparse import SparseTensor, to_dense, reorder


class VocabBuilder(object):
    """
    Class is dessigned to tokenize the sentences of a given text. Also, it
    separates the text from its labels.
    """

    def __init__(self, path_csv_file, vocab_size, training_percentage=0.9, window_size=5):
        """
        Contructor
        :param path_csv_file:
        :param vocab_size:
        :param training_percentage:
        :param window_size:
        """
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.df = pd.read_csv(path_csv_file, on_bad_lines='skip')

        # Separate training and testing sets
        self.train_df = self.df.sample(frac=training_percentage)
        self.test_df = self.df.drop(self.train_df.index)

        # Separates the sentences and the labels
        self.train_sentences, self.train_labels = self.train_df['sentence'], self.train_df['label']
        self.test_sentences, self.test_labels = self.test_df['sentence'], self.test_df['label']

        # Model sentences as tf.Dataset
        self.train_text_ds = tf.data.Dataset.from_tensor_slices(self.train_sentences)
        self.test_text_ds = tf.data.Dataset.from_tensor_slices(self.test_sentences)

        # Vocabulary and the sequences of the training and testing sets
        self.train_sequences, self.inverse_vocab = self.vectorize_data(self.train_text_ds)
        self.test_sequences, _ = self.vectorize_data(self.test_text_ds)

    def vectorize_data(self, text_ds):
        """
        Documentation
        :param text_ds:
        """
        vectorization_layer = TextVectorization(standardize='lower_and_strip_punctuation',
                                                max_tokens=self.vocab_size,
                                                output_mode='int',
                                                output_sequence_length=None)
        vectorization_layer.adapt(text_ds.batch(1024))
        inverse_vocab = vectorization_layer.get_vocabulary()
        vectorized_text_ds = text_ds.map(vectorization_layer)
        sequences = vectorized_text_ds.as_numpy_iterator()
        return sequences, inverse_vocab

    def coocurrence_matrix(self, sequences):
        """
        DOCUMENTATION
        :param sequences:
        """
        pairs = {}
        for seq in tqdm(sequences):
            for target_index in range(len(seq)):
                for window_index in range(1, self.window_size + 1):
                    # target_index + window_index is the index of the context
                    if target_index + window_index < len(seq):
                        # If already exists, add 1/window_index
                        if (seq[target_index], seq[target_index + window_index]) in pairs:
                            pairs[(seq[target_index], seq[target_index + window_index])] += 1 / window_index
                        else:
                            pairs[(seq[target_index], seq[target_index + window_index])] = 1 / window_index
        m_coocurrence_sparse = SparseTensor(indices=[list(key) for key in pairs.keys()],
                                            values=list(pairs.values()),
                                            dense_shape=(len(self.inverse_vocab), len(self.inverse_vocab)))
        m_coocurrence_sparse = tf.sparse.add(reorder(m_coocurrence_sparse),
                                             tf.sparse.transpose(reorder(m_coocurrence_sparse)))
        m_coocurrence = tf.cast(to_dense(m_coocurrence_sparse), dtype=tf.float32)
        return m_coocurrence

    def train_coocurrence_matrix(self):
        """
        Returns the coocurrence matrix of the training set
        """
        return self.coocurrence_matrix(self.train_sequences)

    def test_coocurrence_matrix(self):
        """
        Returns the coocurrence matrix of the testing set
        """
        return self.coocurrence_matrix(self.test_sequences)


if __name__ == '__main__':
    path = 'C:/Users/sfino/OneDrive/Documents/Data Science/NLP/Sentiment_analysis/data/IMDB_Dataset.csv'
    vb = VocabBuilder(path, 1024)
    print(vb.train_coocurrence_matrix())
