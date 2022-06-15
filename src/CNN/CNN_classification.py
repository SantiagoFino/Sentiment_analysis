import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Concatenate, Embedding, Flatten
import numpy as np

class CnnSentenceClassificator(tf.keras.Model):
    """
    Models a 2-layer CNN for sentence classification
    """

    def __init__(self, embedding_dimension, number_of_outputs, seq_max_len, target_embedding, context_embbeding):
        super(CnnSentenceClassificator, self).__init__()
        """
        Constructor
        :param embedding_dimension:
        :param number_of_outputs:
        :param seq_max_len: 
        :param random_embedding:
        :param target_embedding:
        :param context_embbeding:
        """
        self.embedding_dimension = embedding_dimension
        self.number_of_outputs = number_of_outputs
        self.seq_max_len = seq_max_len
        self.target_embedding = target_embedding
        self.context_embedding = context_embbeding

        # CNN
        self.embedding = Embedding(input_dim=4096, output_dim=128, input_length=50, trainable=False)
        # Creates convolutions with different kernel sizes
        '''
        self.convolutions, kernel_sizes = [], [2, 3, 4]
        for kernel_size in kernel_sizes:
            conv = Conv1D(filters=200, kernel_size=kernel_size,
                          padding='same', activation='relu',
                          bias_initializer=tf.keras.initializers.random_uniform,
                          kernel_regularizer=tf.keras.regularizers.L1(l1=0.01),
                          input_shape=(self.seq_max_len, self.embedding_dimension))
            self.convolutions.append(conv)'''
        self.conv1 = Conv1D(filters=200, kernel_size=3,
                            padding='same', activation='relu',
                            bias_initializer=tf.keras.initializers.random_uniform,
                            kernel_regularizer=tf.keras.regularizers.L1(l1=0.01),
                            input_shape=(self.seq_max_len, self.embedding_dimension))
        self.pool1 = MaxPooling1D(pool_size=2)
        self.flat = Flatten()
        self.dense = Dense(units=200,
                           activation='relu',
                           bias_initializer=tf.keras.initializers.random_uniform)
        self.out = Dense(units=self.number_of_outputs,
                         activation='sigmoid')

    def call(self, input_data):
        """
        Defines the function of the CNN
        :param input_data:
        :return: Function describing the CNN
        """
        embedding = self.embedding(input_data)
        # self.convolutions = [conv(embedding) for conv in self.convolutions]
        # f = Concatenate()(self.convolutions)
        f = self.conv1(embedding)
        f = self.pool1(f)
        f = self.flat(f)
        f = self.dense(f)
        f = self.out(f)
        return f


def main_CNN_classification(input_data_train, input_data_test, embedding_dimension, number_of_outputs,
                            seq_max_len, target_embedding, context_embbeding, epochs=10):
    """
    Defines the optimizer, the losses, the accuracy mesurements and trains a model
    :param epochs:
    :param input_data_train:
    :param input_data_test:
    :param embedding_dimension:
    :param number_of_outputs:
    :param seq_max_len:
    :param target_embedding:
    :param context_embbeding:
    """
    # Defines the variables for the compilation of the model
    opt = tf.keras.optimizers.RMSprop()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    model: CnnSentenceClassificator = CnnSentenceClassificator(embedding_dimension=embedding_dimension,
                                                               number_of_outputs=number_of_outputs,
                                                               seq_max_len=seq_max_len,
                                                               target_embedding=target_embedding,
                                                               context_embbeding=context_embbeding)

    # Defines how the backpropagation will work
    @tf.function
    def train_step(input_sentences, labels):
        with tf.GradientTape() as tape:
            predictions = model(input_sentences, training=True)
            tr_loss = loss(labels, predictions)
        gradients = tape.gradient(tr_loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(tr_loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(input_sentences, labels):
        predictions = model(input_sentences, training=False)
        t_loss = loss(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    # Running the model
    for epoch in range(epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        # Training
        for train_seq, train_labels in input_data_train:
            train_step(train_seq, train_labels)

        # Testing
        for test_seq, test_labels in input_data_test:
            test_step(test_seq, test_labels)

        print(f'Epoch {epoch + 1}, '
              f'Loss: {train_loss.result()}, '
              f'Accuracy: {train_accuracy.result() * 100}, '
              f'Test Loss: {test_loss.result()}, '
              f'Test Accuracy: {test_accuracy.result() * 100}')


# %%
if __name__ == '__main__':
    from tensorflow.keras.datasets import imdb
    (x_train, y_train), (x_test, y_test) = imdb.load_data()

