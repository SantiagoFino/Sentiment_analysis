# %%
import pandas as pd
import numpy as np

from Levenshtein import distance


def top_similar_words_by_Levenshtein(word: str, possible_corrections: list, num_words: int, threshole=5):
    """
    Given a word, computes the Levenshtein distance with a list of words and select the most similar ones
    :param word:
    :param possible_corrections:
    :param num_words:
    :param threshole:
    """
    if word == '':
        return {}
    else:
        # Number of possible corrections
        num_samples = len(possible_corrections)
        # Distance from the word to each one of the
        distances = list(map(distance, [word] * num_samples, possible_corrections))

        # DataFrame with the distances btween word and each one of the corrections
        df_distances = pd.DataFrame({'word': possible_corrections, 'distance': distances})
        # corrections which distance is lower than the threshole
        df_distances = df_distances[df_distances['distance'] <= threshole]
        df_distances = df_distances.sort_values(by='distance', ascending=True).head(num_words)
        return dict(zip(list(df_distances['word']), list(df_distances['distance'])))


def generate_context(line: str, targets: list, context_len: int):
    """
    Finds the context of every mistake in the database
    """
    dict_context = {}
    for mistake in targets:
        context = line.split(mistake)
        b_context = context[0].split()[-context_len:] if len(context[0]) > context_len else context[0].split()
        a_context = context[1].split()[:context_len] if len(context[1]) > context_len else context[1].split()
        dict_context[mistake] = (b_context + a_context).remove('') if '' in (b_context + a_context) else b_context + a_context
    return dict_context


class SimilarityMatcher(object):
    def __init__(self, dictionary: list, path_raw_text: str, inverse_vocab: list, target_embedding, context_embedding):
        """
        Constructor
        """
        self.dictionary = set(dictionary)
        self.raw_text = path_raw_text
        self.inverse_vocab = inverse_vocab
        self.target_embedding = target_embedding
        self.context_embedding = context_embedding
        self.probability_matrix = None
        self.mistakes_data_frame = pd.DataFrame()

    def mistake_finder(self):
        """
        Find the words of the text that does not match with any of the dictionary words
        """
        with open(self.raw_text, 'r') as file:
            lines = file.read().splitlines()
            mistakes = []
            indicator = []
            for line in lines:
                words = set(line.split())
                mistakes.append(list(words.difference(self.dictionary)))
                indicator.append(False if len(list(words.difference(self.dictionary))) == 0 else True)
            file.close()
        self.mistakes_data_frame['lines'] = lines
        self.mistakes_data_frame['mistakes'] = mistakes
        self.mistakes_data_frame['exists_mistake'] = indicator

    def mitake_finder(self, n=5):
        """
        Find the words of the text that does not match with any of the dictionary words
        """
        print("...Finding Mistakes...")
        self.lines = set(self.lectura_dicc())
        self.raw_text['errors'] = self.raw_text['Text_Clean'].applymap(lambda x: list(set(x.split()) - self.lines))
        print("...Finding corrections...")
        self.raw_text['corrections'] = self.raw_text['errors'].apply(lambda x: self.top_similar_words_by_Levenshtein(x, n))
        self.raw_text['context'] = self.raw_text.apply(lambda x: self.contextos(x) if len(x['errors']) > 0 else [],
                                                       axis=1)

    def contextos(self, x):
        Lista_contextos = []
        for e in x['errors']:
            # breakpoint()
            if (x['Text_Clean'].partition(e)[0] != '') & (x['Text_Clean'].partition(e)[2] != ''):
                l = [x['Text_Clean'].partition(e)[0].split()[-1], x['Text_Clean'].partition(e)[2].split()[0]]
            elif x['Text_Clean'].partition(e)[0] != '':
                l = [x['Text_Clean'].partition(e)[0].split()[-1], '']
            elif x['Text_Clean'].partition(e)[2] != '':
                l = ['', x['Text_Clean'].partition(e)[2].split()[0]]
            else:
                l = ['', '']
            Lista_contextos.append(l)
        return Lista_contextos

    def generate_mistake_context(self, window_size: int):
        """
        Computes the contexts of each one of the mistakes found in the text
        :param window_size:
        """
        if len(self.mistakes_data_frame) == 0:
            self.mistake_finder()
        contexts = map(generate_context, self.mistakes_data_frame['lines'], self.mistakes_data_frame['mistakes'],
                       [window_size]*len(self.mistakes_data_frame.index))
        return list(contexts)

    def generate_candidates_from_word_dict(self, num_candidates: int):
        """

        """
        if len(self.mistakes_data_frame) == 0:
            self.mistake_finder()
        mistakes = self.mistakes_data_frame['mistakes'].sum(axis=0)
        corrections = map(top_similar_words_by_Levenshtein,
                          mistakes,
                          [list(self.dictionary)]*len(mistakes),
                          [num_candidates]*len(mistakes))
        return dict(zip(mistakes, list(corrections)))

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

    def mean_probabilities(self, candidate_words: list, context_words: list):
        """
        Sum the distribution vectors given in the conditional probability matrix of a given list of words
        :param candidate_words:
        :param context_words:
        :return final_distribution:
        """
        if self.probability_matrix is None:
            self.conditional_probability_matrix()

        # Filters the info of the words/candidates that are not in the inverse_vocab
        no_info_cand_words = [word for word in candidate_words if word not in self.inverse_vocab]
        no_info_cont_words = [word for word in context_words if word not in self.inverse_vocab]
        if len(no_info_cand_words) > 0 and len(no_info_cont_words) > 0:
            print(f'Theres no enough info in the inverse vocab about {no_info_cand_words}')
            print(f'Theres no enough info in the inverse vocab about {no_info_cont_words}')
        clean_candidate_words = list(set(candidate_words) - set(no_info_cand_words))
        clean_context_words = list(set(context_words) - set(no_info_cont_words))

        # Select the columns corresponding to the possible words
        distributions = self.probability_matrix[clean_candidate_words]

        # Select the contexts
        distributions = distributions.loc[clean_context_words]
        probability_mean = distributions.mean(axis=0).T.sort_values().head(1)
        return probability_mean.index[0] if len(probability_mean) > 0 else candidate_words[0]


def analize_text(dictionary: list, path_raw_text: str, inverse_vocab: list, target_embedding, context_embedding,
                 embedding_before_lev=False, window_size=1):
    if embedding_before_lev:
        pass
    else:
        sm = SimilarityMatcher(dictionary=dictionary,
                               path_raw_text=path_raw_text,
                               inverse_vocab=inverse_vocab,
                               target_embedding=target_embedding,
                               context_embedding=context_embedding)

        # Finds the mistakes of the input text
        sm.mistake_finder()

        # Finds the contexts and the possible corrections of each one of the mistakes
        contexts = sm.generate_mistake_context(window_size=window_size)
        candidates = sm.generate_candidates_from_word_dict(num_candidates=3)
        line_corrections = []
        for line in contexts:
            corrections = []
            for mistake in line:
                context = line[mistake]
                candidate = list(candidates[mistake].keys())
                correct_word = sm.mean_probabilities(candidate, context)
                corrections.append(correct_word)
            line_corrections.append(corrections)
        sm.mistake_finder()
        sm.mistakes_data_frame['corrections'] = pd.Series(line_corrections)
        return sm.mistakes_data_frame


# %%
if __name__ == '__main__':
    from GloVe import main_GloVe

    path = 'C:/Users/sfino/OneDrive/Documents/Data Science/NLP/Sentiment_analysis/data/'
    text = path + 'no_stopwords_IMDB.txt'
    error_text = path + 'CleanSpellingErrorText_small.txt'

    inv_vocab, t_weights, c_weights, losses = main_GloVe(path_file=text,
                                                         window_size=5,
                                                         vocab_size=4000,
                                                         iteraciones=1000)
    # %%
    with open(path + 'CleanEnglishDict.txt', 'r') as r:
        dic = r.read().splitlines()
        r.close()

    spelling_corrector = SimilarityMatcher(dictionary=dic,
                                           path_raw_text=error_text,
                                           inverse_vocab=inv_vocab,
                                           target_embedding=t_weights,
                                           context_embedding=c_weights)

    line_corr = analize_text(dic, error_text, inv_vocab, t_weights, c_weights)
