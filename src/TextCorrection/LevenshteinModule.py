import pandas as pd
import numpy as np
from Levenshtein import distance


class SimilarityMatcher(object):
    def __init__(self, dictionary, path_raw_text, inverse_vocab, target_embedding, context_embedding):
        """
        Constructor
        """
        self.dictionary = set(dictionary)
        self.raw_text = path_raw_text
        self.inverse_vocab = inverse_vocab
        self.target_embedding = target_embedding
        self.context_embedding = context_embedding
        self.mistakes = set()
        self.probability_matrix = None

    def mistake_finder(self):
        """
        Find the words of the text that does not match with any of the dictionary words
        """
        print("...Finding Mistakes...")
        with open(self.raw_text, 'r') as file:
            lines = file.read().splitlines()
            for line in lines:
                words = set(line.split())
                self.mistakes.add(words - self.dictionary)

    def conditional_probability_matrix(self):
        """
        Computes the conditional probability matrix given certain embeddings
        :return df_matrz_probabilidades:
        """
        df_target_embedding = pd.DataFrame(self.target_embedding,
                                           index=self.inverse_vocab).loc[self.inverse_vocab].to_numpy()
        df_context_embedding = pd.DataFrame(self.context_embedding,
                                            index=self.inverse_vocab).loc[self.inverse_vocab].to_numpy()

        partial = np.exp(np.matmul(df_context_embedding, df_target_embedding.T))
        matrix_probabilidades = partial / partial.sum(axis=0, keepdims=True)
        df_matrix_probabilidades = pd.DataFrame(data=matrix_probabilidades, index=self.inverse_vocab,
                                                columns=self.inverse_vocab)
        self.probability_matrix = df_matrix_probabilidades

    def top_similar_words_by_Levenshtein(self, word, possible_corrections, n):
        """
        Given a word, computes the Levenshtein distance with a list of words and select the most similar ones
        """
        return self.inverse_vocab + word + possible_corrections + n

    def sum_probabilities(self, candidate_words: list, context_words: list):
        """
        Sum the distribution vectors given in the conditional probability matrix of a given list of words
        :param candidate_words:
        :param context_words:
        :return final_distribution:
        """
        if self.probability_matrix is None:
            self.conditional_probability_matrix()

        # Select the columns corresponding to the possible words
        distributions = self.probability_matrix[candidate_words]

        # Select the contexts
        distributions = distributions.iloc[context_words]
        probability_mean = distributions.mean(axis=0).T
        return probability_mean.sort_values()
