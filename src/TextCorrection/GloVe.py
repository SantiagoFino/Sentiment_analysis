import os

import pandas as pd
import tqdm

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from DataProcessor import VocabBuilder

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


class GloVe(tf.keras.Model):
    """
    Modela el algoritmo de NLP Global Vectors for Word Representation
    """

    def __init__(self, vocab_size, embedding_dimension=128, exp_alpha=0.75, x_max=100):
        super(GloVe, self).__init__()
        """
        Constructor
        Args:
            embedding_dimension: int, Dimension de los embedimientos de los target y los context
            vocab_size: int, tamano del vocabulario
            matriz_coocurrencias: EagerTensor, tensor con las frecuencias de las parejas target-context
            exp_alpha: float, exponente alpha de la funcion f
            x_max: int, maxima frecuencia permitida
        """
        self.vocab_size = vocab_size
        self.exp_alpha = exp_alpha
        self.x_max = x_max

        # Embedimientos del target y del context
        self.target_embedding = layers.Embedding(vocab_size, embedding_dimension)
        self.context_embedding = layers.Embedding(vocab_size, embedding_dimension)

        # Vectores de bias de la funcion de perdida inicializados aleatoriamente shape = (vocab_size, 1)
        self.target_bias = tf.Variable(tf.random.normal(shape=(vocab_size, 1),
                                                        mean=0), trainable=True)
        self.context_bias = tf.Variable(tf.random.normal(shape=(vocab_size, 1),
                                                         mean=0), trainable=True)

        # Funcion auxiliar para el costo
        self.f_techo = lambda x: tf.clip_by_value(tf.pow((x / self.x_max), self.exp_alpha), clip_value_min=0,
                                                  clip_value_max=1)

    def call(self, matriz_coocurrencias):
        """
        Funcion de perdida
        Returns:
            perdida: float, perdida del modelo
        """
        # Matrices correspondientes de los embeddings del target y del context shape = (vocab_size, emb_dim)
        target_matrix = self.target_embedding(tf.range(0, self.vocab_size))
        context_matrix = self.context_embedding(tf.range(0, self.vocab_size))

        # Producto punto entre las matrices de los embeddings: <W_t, W_c^T>^T
        dots = tf.transpose(tf.matmul(target_matrix, tf.transpose(context_matrix)))

        # Funcion de perdida
        diferencia = (dots + tf.transpose(self.target_bias) + self.context_bias - tf.math.log(
            1 + matriz_coocurrencias)) ** 2
        perdida = tf.reduce_sum(tf.math.multiply(self.f_techo(matriz_coocurrencias), diferencia))
        return perdida


def main_GloVe(path_file, window_size, vocab_size, iteraciones=2000):
    """
    DOCUMENTACION
    Args:
        path_file: Ruta del dataset
        window_size: tamano de la ventana del contexto
        vocab_size: numero de palabras en el vocabulario
        iteraciones: int, cantidad de iteraciones que realizara el optimizador
    Returns:
        target_weights, context_weights, t_bias, c_bias, inverse_vocab, perdidas
    """
    # Construccion del vocabulario
    vocab_buider = VocabBuilder(path_file=path_file,
                                window_size=window_size,
                                vocab_size=vocab_size)

    # Construccion de la matriz de coocurrencias
    matriz_coocurrencias = vocab_buider.build_matriz_coocurrencias()
    vocab_size_cp = len(vocab_buider.inverse_vocab)
    # Definicion del modelo
    glove = GloVe(vocab_size=vocab_size_cp)

    # Agrego los bias como nuevos pesos
    glove.trainable_weights.append(glove.target_bias)
    glove.trainable_weights.append(glove.context_bias)

    opt_descenso = tf.keras.optimizers.Nadam()
    var_list_fn = lambda: glove.trainable_weights
    loss_fn = lambda: glove.call(matriz_coocurrencias)

    # Optimizacion
    perdidas = []
    for _ in tqdm.tqdm(range(iteraciones)):
        opt_descenso.minimize(loss=loss_fn, var_list=var_list_fn)
        perdidas.append(loss_fn().numpy())

    # Vocabulario ordenado
    inverse_vocab = vocab_buider.vectorization_layer.get_vocabulary()
    # Pesos de los embeddings
    target_weights = np.array(glove.trainable_weights[0])
    context_weights = np.array(glove.trainable_weights[1])

    return inverse_vocab, target_weights, context_weights, perdidas


if __name__ == "__main__":
    print('siuuuuu')
