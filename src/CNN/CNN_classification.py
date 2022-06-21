import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Concatenate, Embedding, Flatten

from DataProcessor import VocabBuilder
from GloVe import main_GloVe


class CnnSentenceClassificator(tf.keras.Model):
    """
    Models a 2-layer CNN for sentence classification
    """

    def __init__(self, number_of_outputs, seq_max_len, embedding_weights, embedding_dimension=128):
        super(CnnSentenceClassificator, self).__init__()
        """
        Constructor
        :param embedding_dimension:
        :param number_of_outputs:
        :param seq_max_len:
        :param embedding_weights:
        """
        self.embedding_dimension = embedding_dimension
        self.number_of_outputs = number_of_outputs
        self.seq_max_len = seq_max_len
        self.embedding_weights = embedding_weights

        # CNN
        self.embedding = Embedding(input_dim=1024, output_dim=128, input_length=50, trainable=False)
        self.embedding.set_weights(embedding_weights)
        # Creates convolutions with different kernel sizes
        self.convolutions, kernel_sizes = [], [2, 3, 4]
        for kernel_size in kernel_sizes:
            conv = Conv1D(filters=200, kernel_size=kernel_size,
                          padding='same', activation='relu',
                          bias_initializer=tf.keras.initializers.random_uniform,
                          kernel_regularizer=tf.keras.regularizers.L1(l1=0.01),
                          input_shape=(self.seq_max_len, self.embedding_dimension))
            self.convolutions.append(conv)
        self.pool1 = MaxPooling1D(pool_size=2)
        self.flat = Flatten()
        self.dense = Dense(units=200,
                           activation='relu',
                           bias_initializer=tf.keras.initializers.random_uniform)
        self.out = Dense(units=self.number_of_outputs,
                         activation='softmax')

    def __call__(self, input_data, *args, **kwargs):
        """
        Defines the function of the CNN
        :param input_data:
        :return: Function describing the CNN
        """
        embedding = self.embedding(input_data)
        self.convolutions = [conv(embedding) for conv in self.convolutions]
        f = Concatenate()(self.convolutions)
        f = self.pool1(f)
        f = self.flat(f)
        f = self.dense(f)
        f = self.out(f)
        return f


def main_CNN_classification(input_data_train, input_data_test, embedding_dimension, number_of_outputs,
                            seq_max_len, embedding_weights, epochs=10):
    """
    Defines the optimizer, the losses, the accuracy mesurements and trains a model
    :param epochs:
    :param input_data_train:
    :param input_data_test:
    :param embedding_dimension:
    :param number_of_outputs:
    :param seq_max_len:
    :param embedding_weights:
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
                                                               embedding_weights=embedding_weights)

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
              f'Accuracy: {train_accuracy.result() * 100},'
              f'Test Loss: {test_loss.result()}, '
              f'Test Accuracy: {test_accuracy.result() * 100}')


if __name__ == '__main__':
    path = 'C:/Users/sfino/OneDrive/Documents/Data Science/NLP/Sentiment_analysis/data/IMDB_Dataset.csv'
    vb = VocabBuilder(path_csv_file=path,
                      vocab_size=1024)
    (training_sentences, training_labels) = vb.train_sequences, vb.train_labels
    (test_sentences, testing_labels) = vb.test_sequences, vb.test_labels

    weights, _, _ = main_GloVe(path_file=path,
                               vocab_size=1024,
                               iterations=10)

    main_CNN_classification(input_data_train=(training_sentences, training_labels),
                            input_data_test=(test_sentences, testing_labels),
                            embedding_dimension=128,
                            number_of_outputs=2,
                            seq_max_len=200,
                            embedding_weights=weights)
