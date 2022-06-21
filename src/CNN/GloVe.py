from tqdm import tqdm

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras import Model

from DataProcessor import VocabBuilder


class GloVe(Model):
    """
    Performs the GloVe algorithm described in Pennington, et al (2014)
    """

    def __init__(self, vocab_size, embedding_dimension=128, exp_alpha=0.75, x_max=100):
        """
        Constructor
        :param vocab_size:
        :param embedding_dimension:
        :param exp_alpha:
        :param x_max:
        """
        super(GloVe, self).__init__()

        self.vocab_size = vocab_size
        self.exp_alpha = exp_alpha
        self.x_max = x_max

        # Embedimientos del target y del context
        self.target_embedding = Embedding(vocab_size, embedding_dimension)
        self.context_embedding = Embedding(vocab_size, embedding_dimension)

        # Vectores de bias de la funcion de perdida inicializados aleatoriamente shape = (vocab_size, 1)
        self.target_bias = tf.Variable(tf.random.normal(shape=(vocab_size, 1),
                                                        mean=0), trainable=True)
        self.context_bias = tf.Variable(tf.random.normal(shape=(vocab_size, 1),
                                                         mean=0), trainable=True)

        # Funcion auxiliar para el costo
        self.f_techo = lambda x: tf.clip_by_value(tf.pow((x / self.x_max), self.exp_alpha), clip_value_min=0,
                                                  clip_value_max=1)

    def __call__(self, coocurrence_matrix,  *args, **kwargs):
        """
        Funcion de perdida
        Returns:
            loss:
        """
        # Matrices correspondientes de los embeddings del target y del context shape = (vocab_size, emb_dim)
        target_matrix = self.target_embedding(tf.range(0, self.vocab_size))
        context_matrix = self.context_embedding(tf.range(0, self.vocab_size))

        # dot probuct between the embeddings: <W_t, W_c^T>^T
        dots = tf.transpose(tf.matmul(target_matrix, tf.transpose(context_matrix)))

        # loss function
        partial = (dots + tf.transpose(self.target_bias) + self.context_bias - tf.math.log(
                   1 + coocurrence_matrix)) ** 2
        loss = tf.reduce_sum(tf.math.multiply(self.f_techo(coocurrence_matrix), partial))
        return loss


def main_GloVe(path_file, vocab_size, iterations=1000, training_weights=True):
    """
    DOCUMENTACION
    Args:
        path_file: Ruta del dataset
        vocab_size: numero de palabras en el vocabulario
        iterations: int, cantidad de iteraciones que realizara el optimizador
        training_weights:
    Returns:
        target_weights, context_weights, t_bias, c_bias, inverse_vocab, perdidas
    """
    # Construccion del vocabulario
    vocab_buider = VocabBuilder(path_csv_file=path_file,
                                vocab_size=vocab_size)

    # Coocurrence matrix
    matriz_coocurrencias = vocab_buider.train_coocurrence_matrix() if training_weights \
                                                                   else vocab_buider.test_coocurrence_matrix()
    vocab_size_cp = len(vocab_buider.inverse_vocab)

    # Model definition
    glove = GloVe(vocab_size=vocab_size_cp)

    # Add bias weights
    glove.trainable_weights.append(glove.target_bias)
    glove.trainable_weights.append(glove.context_bias)

    # Default optimizer: Nadam
    opt = tf.keras.optimizers.Nadam()
    var_list_fn = lambda: glove.trainable_weights
    loss_fn = lambda: glove(matriz_coocurrencias)

    # Backpropagation
    losses = []
    for _ in tqdm(range(iterations)):
        opt.minimize(loss=loss_fn, var_list=var_list_fn)
        losses.append(loss_fn().numpy())

    # Embeddings weights
    target_weights = glove.trainable_weights[0]
    context_weights = glove.trainable_weights[1]

    return target_weights, context_weights, losses


if __name__ == '__main__':
    path = 'C:/Users/sfino/OneDrive/Documents/Data Science/NLP/Sentiment_analysis/data/IMDB_Dataset.csv'
    main_GloVe(path_file=path,
               vocab_size=1024,
               iterations=10,
               training_weights=True)
