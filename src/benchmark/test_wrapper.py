import os
from typing import List, Tuple, Any
from src.decision_templates import DecisionTemplatesClassifier
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.externals.joblib import load
from sklearn.exceptions import NotFittedError
from src.settings import PICKLED_RANDOM_SUBSETS
from copy import deepcopy

import numpy as np


class DecisionTemplatesClassifierWrapper(DecisionTemplatesClassifier):

    def __init__(self, fold, cv, learning_set, estimators: List[Tuple[str, BaseEstimator]],
                 groups_mapping: List[Tuple[Any]] = None, similarity_measure: str = 'euclidean',
                 template_construction: str = 'avg', template_fit_strategy: str = 'one_per_class',
                 n_templates: int = 1, decision_strategy: str = 'most_similar_template',
                 similarity_for_group: str = 'separately', k_similar_templates: int = 1, n_jobs: int = 1) -> None:

        super().__init__(estimators, groups_mapping, similarity_measure, template_construction, template_fit_strategy,
                         n_templates, decision_strategy, similarity_for_group, k_similar_templates, n_jobs)

        self.fold = fold
        self.cv = cv
        self.learning_set = learning_set

    def _fit_estimators(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray):
        temporary_estimators = [deepcopy(clf) for _, clf in self.estimators]
        try:
            for clf in temporary_estimators:
                check_is_fitted(clf, 'classes_')
        except NotFittedError:
            print("Not Fitted Error")
            temporary_estimators = super()._fit_estimators(X, y, sample_weight)

        return temporary_estimators

    def make_weights(self, n_samples: int,  sample_weight: np.ndarray):
        path = os.path.join(PICKLED_RANDOM_SUBSETS, "cv_" + str(self.cv) + "_fold_" + str(self.fold + 1)
                            + "_" + str(self.learning_set) + "_" + self.template_fit_strategy
                            + "_" + str(self.n_templates) + ".pickle")

        return load(path)