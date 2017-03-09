import os
from src.decision_templates import DecisionTemplatesClassifier
from src.benchmark.k_fold_estimator_creator import KFoldEstimatorsCreator
from src.benchmark.weights_creator import RandomWeightsCreator
import numpy as np


class DecisionTemplatesClassifierWrapper(DecisionTemplatesClassifier):

    def __init__(self, fold, cv, learning_set, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.learning_set = learning_set
        self.cv = cv
        self.fold = fold
        self.fold_estimator_creator = KFoldEstimatorsCreator(self.learning_set, self.fold, self.cv)
        self.weights_creator = RandomWeightsCreator(self.learning_set, self.fold, self.cv, self.template_fit_strategy,
                                                    self.n_templates)

    def _fit_estimators(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray):

        if os.path.exists(self.fold_estimator_creator.get_path(len(self.estimators))):
            return self.fold_estimator_creator.load_base_estimators(len(self.estimators))
        else:
            temporary_estimators = super()._fit_estimators(X, y, sample_weight)
            self.fold_estimator_creator.save_base_estimators(temporary_estimators)
            return temporary_estimators

    def make_weights(self, n_samples: int, sample_weight: np.ndarray):
        if os.path.exists(self.weights_creator.path):
            return self.weights_creator.load_base_estimators()
        else:
            temporary_weights = list(super().make_weights(n_samples, sample_weight))
            self.weights_creator.save_base_estimators(temporary_weights)
            return temporary_weights