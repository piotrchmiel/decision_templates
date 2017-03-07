import os
import warnings
from copy import deepcopy
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from src.dataset import DataProvider, LearningSet
from sklearn.externals.joblib import dump, load

from sklearn.preprocessing import LabelEncoder
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.fixes import bincount
from sklearn.utils import check_random_state
from sklearn.ensemble.bagging import MAX_INT
from typing import List
import numpy as np
import time

from src.settings import PICKLED_RANDOM_SUBSETS
from src.benchmark.testing_parameters import TESTING_PARAMETERS


class FoldRandomWeightsGenerator(object):
    def __init__(self):
        self.random_state = check_random_state(None)
        self.provider = DataProvider()
        self.to_generate = set()
        for key, value in TESTING_PARAMETERS.items():
            self.to_generate.add((value['template_fit_strategy'], value['n_templates']))
        if not os.path.exists(PICKLED_RANDOM_SUBSETS):
            os.mkdir(PICKLED_RANDOM_SUBSETS)

    def parallel_make_and_save_weights(self, learning_set: LearningSet, cv: int = 10):
        k_fold_iterator = self.provider.get_k_fold_generator(learning_set, cv)

        parallel = Parallel(n_jobs=1, verbose=0)
        parallel(delayed(self.make_and_save_weights)(train_X.shape[0], learning_set, fold, cv)
                 for fold, (train_X, _, _, _) in enumerate(k_fold_iterator))

    def _random_set(self, number_of_class_templates, bootstrap: bool, n_samples: int, sample_weight: List[int]):
        for seed in self.random_state.randint(MAX_INT, size=number_of_class_templates):
            random_state = np.random.RandomState(seed)
            time.sleep(1)
            if bootstrap:
                indices = random_state.randint(0, n_samples, n_samples)
                curr_sample_weight = sample_weight.copy()
                sample_counts = bincount(indices, minlength=n_samples)
                yield curr_sample_weight * sample_counts
            else:
                yield random_state.randint(0, 2, n_samples)

    def make_weights(self, strategy, n_templates, n_samples: int, sample_weight: np.ndarray):
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,))
        else:
            curr_sample_weight = sample_weight.copy()

        if strategy == 'one_per_class':
            return curr_sample_weight,
        elif strategy == 'bootstrap':
            return self._random_set(number_of_class_templates=n_templates, bootstrap=True,
                                    n_samples=n_samples, sample_weight=curr_sample_weight)
        elif strategy == 'random_subspace':
            return self._random_set(number_of_class_templates=n_templates, bootstrap=False,
                                    n_samples=n_samples, sample_weight=curr_sample_weight)


    def save_weight(self, value, learning_set, fold, cv, strategy, n_templates):
        path = os.path.join(PICKLED_RANDOM_SUBSETS, "cv_" + str(cv) + "_fold_" + str(fold + 1)
                            + "_" + str(learning_set) + "_" + strategy + "_" + str(n_templates) + ".pickle")
        return dump(value, path)

    def make_and_save_weights(self, n_samples, learning_set, fold, cv):
        for strategy, n_templates in self.to_generate:
            weights = list(self.make_weights(strategy, n_templates, n_samples, None))
            self.save_weight(weights, learning_set, fold, cv, strategy, n_templates)

if __name__ == '__main__':

    estimator_creator = FoldRandomWeightsGenerator()
    estimator_creator.parallel_make_and_save_weights(LearningSet.segment)
