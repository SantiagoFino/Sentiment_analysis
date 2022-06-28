# %%
"""Define el procesamiento y limpieza de los datos que seran ajustados en el modelo"""
import re
import string
import os
from tqdm import tqdm
import warnings
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.sparse import SparseTensor, to_dense, reorder


class VocabBuilder(object):
    def __init__(self, path_file, window_size, vocab_size, negative_samples_per_positive=0, window_size_salud=500):
        """
        Constructor
        Args:
            path_file:
            window_size:
            vocab_size:
            negative_samples_per_positive:
        """
        self.path_to_file = path_file
        self.tf_training_data = None
        self.window_size = window_size
        self.negative_samples_per_positive = negative_samples_per_positive
        self.vocab_size = vocab_size
        self.window_size_salud = window_size_salud
        self.text_ds = tf.data.TextLineDataset(self.path_to_file)
        self.text_ds = self.text_ds.filter(lambda x: tf.cast(tf.strings.length(x), bool))
        self.text_ds_iterator = self.text_ds.as_numpy_iterator()
        self.maximum_sequence_length = int(np.max([len(f.split()) for f in self.text_ds_iterator]))
        self.vectorization_layer = keras.layers.TextVectorization(
            standardize='lower_and_strip_punctuation',
            max_tokens=vocab_size,
            output_mode='int',
            output_sequence_length=self.maximum_sequence_length)

        self.vectorization_layer.adapt(self.text_ds.batch(1024))
        self.inverse_vocab = self.vectorization_layer.get_vocabulary()
        self.vectorized_text_ds = self.text_ds.map(self.vectorization_layer)
        self.sequences = self.vectorized_text_ds.as_numpy_iterator()

    def build_matriz_coocurrencias(self):
        """
        Crea una matriz de coocurrencias
        Arge:
            glove: bool, True si la matriz de frecuencias va a ser utilizada por el algoritmo GloVe
            target_unico: bool, True si el target es unicamente self.sequences[0] (caso CUPs, CUMs e insumos)
            peso_por_longitud: bool, True si se le va asignar mas peso a aquellas secuencias mas cortas
        Returns:
            m_coocurrencias: tensor que representa la matriz de frecuencias
        """
        parejas = {}
        for seq in tqdm(self.sequences):
            # Los targets de una secuencia son todas las palabras
            for target_index in range(len(seq)):
                for window_index in range(1, self.window_size + 1):
                    # Note que target_index + window_index es el indice del context
                    if target_index + window_index < len(seq):
                        # Si ya existe, sume 1/window_index
                        if (seq[target_index], seq[target_index + window_index]) in parejas:
                            parejas[(seq[target_index], seq[target_index + window_index])] += 1 / window_index
                        else:
                            parejas[(seq[target_index], seq[target_index + window_index])] = 1 / window_index
        m_coocurrencias_sparse = SparseTensor(indices=[list(key) for key in parejas.keys()],
                                              values=list(parejas.values()),
                                              dense_shape=(len(self.inverse_vocab), len(self.inverse_vocab)))
        m_coocurrencias_sparse = tf.sparse.add(reorder(m_coocurrencias_sparse),
                                               tf.sparse.transpose(reorder(m_coocurrencias_sparse)))
        m_coocurrencias = tf.cast(to_dense(m_coocurrencias_sparse), dtype=tf.float32)
        return m_coocurrencias


def remove_caps_punct(text: str):
    """
    Removes capital letters and punctuation from a textline
    """
    text_nopunct = text.translate(str.maketrans('', '', string.punctuation))
    clean_text = [word.lower() for word in text_nopunct.split()]
    return ' '.join(clean_text)


class DataCleaning(object):
    """
    Class Description
    """
    def __init__(self, path_text: str, stop_words: list, path_output_file: str):
        """
        Constructor
        :param path_text:
        :param stop_words:
        :param path_output_file:
        """
        self.text = path_text
        self.stop_words = stop_words
        self.output_file = path_output_file

    def remove_stop_words(self, line: str):
        """
        Drops the stop words out of a str line
        :param line:
        :returns:
        """
        no_stopwords = [word for word in line.split() if word not in self.stop_words]
        return ' '.join(no_stopwords)

    def clean_file(self):
        # Reading the input file
        with open(self.text, 'r') as file:
            lines = file.read().splitlines()
            no_caps = map(remove_caps_punct, lines)
            no_stopwords = map(self.remove_stop_words, no_caps)

        # Writting the clean file
        with open(self.output_file, 'w') as r:
            r.write('\n'.join(no_stopwords))


if __name__ == "__main__":
    path = 'C:/Users/sfino/OneDrive/Documents/Data Science/NLP/Sentiment_analysis/data/'
    input_file = path + 'dicc_ingles.txt'
    output_file = path + 'CleanEnglishDict.txt'
    s_words = []

    dc = DataCleaning(path_text=input_file,
                      stop_words=s_words,
                      path_output_file=output_file)

