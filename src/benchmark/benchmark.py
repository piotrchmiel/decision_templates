import os
from typing import List
from copy import deepcopy
import numpy as np

from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection._validation import _score

from src.dataset import DataProvider, LearningSet
from src.benchmark.test_wrapper import DecisionTemplatesClassifierWrapper
from src.benchmark.create_k_fold_estimators import KFoldEstimatorsCreator


class Benchmark(object):
    def __init__(self):
        self.provider = DataProvider()
        self.estimator_creator = KFoldEstimatorsCreator()

    def cross_validation_score(self, parameters, learning_set: LearningSet, cv: int = 10, n_jobs: int = 1,
                               verbose: int = 0) -> List[float]:

        k_fold_iterator = self.provider.get_k_fold_generator(learning_set, cv)

        parallel = Parallel(n_jobs=n_jobs, verbose=verbose)

        scores = parallel(delayed(self._fit_and_score)(deepcopy(parameters), fold, cv, learning_set,
                                                       train_X, train_y, test_X, test_y)
                          for fold, (train_X, train_y, test_X, test_y) in enumerate(k_fold_iterator))
        return scores


    def _fit_and_score(self, parameters, fold, cv, learning_set, train_X, train_y, test_X, test_y):
        estimators = self.estimator_creator.load_base_estimators(learning_set=learning_set, fold=fold, cv=cv)
        parameters['estimators'] = estimators
        estimator = DecisionTemplatesClassifierWrapper(fold, cv, learning_set, **parameters)
        estimator.fit(train_X, train_y)
        scorer = check_scoring(estimator, scoring=None)
        return _score(estimator, test_X, test_y, scorer)