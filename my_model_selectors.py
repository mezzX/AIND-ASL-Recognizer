import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_score = float('inf')
        best_model = None

        try:
            for num in range(self.min_n_components, self.max_n_components+1):
                model = self.base_model(num)
                logL = model.score(self.X, self.lengths)
                p = (num*(num-1) + 2*len(self.X[0])*num)
                score = -2 * logL + p * np.log(len(self.X))
                if score < best_score:
                    best_score = score
                    best_model = model
            return best_model

        except:
            return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_score = float('-inf')
        best_model = None

        for num in range(self.min_n_components, self.max_n_components+1):
            logP1 = {}
            words = []
            for train in self.words.keys():
                X_train, lengths_train = self.hwords[train]
                try:
                    model = GaussianHMM(num, covariance_type='diag', n_iter=1000,
                                        random_state=self.random_state,
                                        verbose=self.verbose).fit(X_train, lengths_train)
                    logP1[train] = model.score(X_train, lengths_train)
                    words.append(train)

                except ValueError:
                    pass

                try:
                    if len(words) > 1:
                        logP2 = sum([logP1[word] for word in words if word != self.this_word])
                        score = logP1[self.this_word] - (1/(len(words)-1)) * logP2
                        if score > best_score:
                            best_score = score
                            best_model = model

                except KeyError:
                    best_model = model

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_score = float('-inf')
        best_model = None
        kfold_n = min(3, len(self.sequences))

        if kfold_n == 1:
            for num in range(self.min_n_components, self.max_n_components+1):
                model = self.base_model(num)
                try:
                    score = model.score(self.X, self.lengths)
                    if score > best_score:
                        best_score = score
                        best_model = model

                except:
                    pass

        else:
            split_method = KFold(kfold_n)
            for num in range(self.min_n_components, self.max_n_components+1):
                try:
                    log_sum = 0
                    length = 0
                    for cv_train, cv_test in split_method.split(self.sequences):
                        X_train, lengths_train = combine_sequences(cv_train, self.sequences)
                        X_test, lengths_test = combine_sequences(cv_test, self.sequences)
                        model = GaussianHMM(num, covariance_type='diag', n_iter=1000,
                                        random_state=self.random_state,
                                        verbose=self.verbose).fit(X_train, lengths_train)
                        score = model.score(X_test, lengths_test)
                        log_sum += score
                        length += 1

                    avg_score = log_sum / length
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = model

                except:
                    pass

        return best_model