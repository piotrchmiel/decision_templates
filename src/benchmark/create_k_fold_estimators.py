import os
import warnings
from copy import deepcopy
from sklearn.naive_bayes import GaussianNB, BernoulliNB
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

from src.settings import PICKLED_ESTIMATORS_DIR


class KFoldEstimatorsCreator(object):
    def __init__(self):
        self.provider = DataProvider()
        if not os.path.exists(PICKLED_ESTIMATORS_DIR):
            os.mkdir(PICKLED_ESTIMATORS_DIR)
        self.random_state = check_random_state(None)

    def fit_and_save_base_estimators(self, estimators, learning_set: LearningSet, cv: int = 10):
        k_fold_iterator = self.provider.get_k_fold_generator(learning_set, cv)

        parallel = Parallel(n_jobs=1, verbose=0)
        parallel(delayed(self.fit_base_estimators)(fold, learning_set, deepcopy(estimators), train_X, train_y, cv)
                 for fold, (train_X, train_y, _, _) in enumerate(k_fold_iterator))

    def fit_base_estimators(self, fold, learning_set, estimator_tup, X, y, cv):
        le_ = LabelEncoder()
        le_.fit(y)
        transformed_y = le_.transform(y)
        for named_estimator in estimator_tup:
            named_estimator[1].fit(X, transformed_y)

        path = os.path.join(PICKLED_ESTIMATORS_DIR, "cv_" + str(cv) + "_fold_" + str(fold + 1)
                            + "_" + str(learning_set) + ".pickle")
        dump(estimator_tup, path, compress=4)

    def load_base_estimators(self, learning_set, fold, cv):
        path = os.path.join(PICKLED_ESTIMATORS_DIR, "cv_" + str(cv) + "_fold_" + str(fold + 1)
                            + "_" + str(learning_set) + ".pickle")
        return load(path)


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    estimator_creator = KFoldEstimatorsCreator()

    gnb = GaussianNB()
    bnb = BernoulliNB()

    mlp1 = MLPClassifier(max_iter=100, activation='logistic')
    mlp2 = MLPClassifier(max_iter=500, activation='logistic')
    mlp3 = MLPClassifier(max_iter=1000, activation='logistic')

    estimators = [("gnb", gnb), ("bnb", bnb), ("mlp1", mlp1), ("mlp2", mlp2), ("mlp3", mlp3)]

    for learning_set in LearningSet:
        print(learning_set)
        estimator_creator.fit_and_save_base_estimators(estimators=estimators, learning_set=learning_set)

