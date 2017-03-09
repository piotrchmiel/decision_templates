import os

from typing import List
from sklearn.base import BaseEstimator
from src.learning_set import LearningSet
from sklearn.externals.joblib import dump, load
from src.settings import PICKLED_ESTIMATORS_DIR


class KFoldEstimatorsCreator(object):
    def __init__(self, learning_set: LearningSet, fold: int, cv: int):
        if not os.path.exists(PICKLED_ESTIMATORS_DIR):
            os.mkdir(PICKLED_ESTIMATORS_DIR)
        self._learning_set = learning_set
        self._fold = fold
        self._cv = cv
        self._path = os.path.join(PICKLED_ESTIMATORS_DIR, "cv_{0}_fold_{1}_{2}_len_{3}.pickle")

    def save_base_estimators(self, estimators: List[BaseEstimator]):
        dump(estimators, self.get_path(len(estimators)), compress=2)

    def load_base_estimators(self, estimators_size: int):
        return load(self.get_path(estimators_size))

    def get_path(self, estimators_size: int):
        return self._path.format(self._cv, self._fold + 1, str(self._learning_set), estimators_size)



