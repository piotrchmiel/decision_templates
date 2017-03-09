import os
import numpy as np
from typing import List
from sklearn.externals.joblib import load
from sklearn.externals.joblib import dump
from src.learning_set import LearningSet
from src.settings import PICKLED_RANDOM_SUBSETS


class RandomWeightsCreator(object):
    def __init__(self, learning_set: LearningSet, fold: int, cv: int, strategy: str, n_templates: int):
        if not os.path.exists(PICKLED_RANDOM_SUBSETS):
            os.mkdir(PICKLED_RANDOM_SUBSETS)
        self._learning_set = learning_set
        self._fold = fold
        self._cv = cv
        self.path = os.path.join(PICKLED_RANDOM_SUBSETS, "cv_{0}_fold_{1}_{2}_strategy_{3}_{4}.pickle").format(
            learning_set, fold + 1, str(learning_set), strategy, n_templates)

    def save_base_estimators(self, weights: List[np.ndarray]):
        dump(weights, self.path, compress=2)

    def load_base_estimators(self):
        return load(self.path)

