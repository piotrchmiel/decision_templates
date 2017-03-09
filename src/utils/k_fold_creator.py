import os
import numpy as np
from typing import List, Tuple

from sklearn.model_selection import StratifiedKFold
from sklearn.externals.joblib import dump, load

from src.learning_set import LearningSet
from src.settings import DATA_DIR


class DefaultKFoldCreator(object):

    def __init__(self, learning_set: LearningSet, X: np.ndarray, y: np.ndarray, source: str):
        self.X = X
        self.y = y
        self.path = os.path.join(DATA_DIR, source)
        self.filename = "cv_{0}_" + str(learning_set) + ".pickle"
        self.learning_set = learning_set

    def make_k_fold_generator(self, cv: int):
        cv_path = os.path.join(self.path, self.filename.format(cv))
        if not os.path.exists(cv_path):
            k_fold_splits = self.save_folds(cv_path, cv)
        else:
            k_fold_splits = load(cv_path)

        def k_fold_generator(splits: List[Tuple[np.ndarray, np.ndarray]]):
            for train_index, test_index in splits:
                yield self.X[train_index], self.y[train_index], self.X[test_index], self.y[test_index]

        return k_fold_generator(k_fold_splits)

    def save_folds(self, file_path: str, cv: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        skf = StratifiedKFold(n_splits=cv)
        k_fold_split = skf.split(self.X, self.y)
        splits = []
        for train_index, test_index in k_fold_split:
            splits.append((train_index, test_index))
        dump(value=splits, filename=file_path, compress=3)

        return splits
