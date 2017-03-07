import os
from enum import Enum, unique
from typing import Tuple, List
import numpy as np
from src.readers.keel import KeelReader
from sklearn.datasets.mldata import fetch_mldata
from sklearn.model_selection import StratifiedKFold
from src.settings import DATA_DIR
from sklearn.externals.joblib import dump, load


@unique
class LearningSet(Enum):
    abalone = 1
    breast = 2
    cleveland = 3
    dermatology = 4
    ecoli = 5
    flare = 6
    iris = 7
    kddcup = 8
    letter = 9
    mnist = 10
    poker = 11
    satimage = 12
    segment = 13
    shuttle = 14
    vehicle = 15
    vowel = 16
    wine = 17
    yeast = 18


class DataProvider(object):

    def __init__(self):
        self.keel_reader = KeelReader()
        self.source_map = {
            LearningSet.abalone: {'source': 'keel', 'filename': 'abalone', 'missing_values': False},
            LearningSet.breast: {'source': 'keel', 'filename': 'breast', 'missing_values': True},
            LearningSet.cleveland: {'source': 'keel', 'filename': 'cleveland', 'missing_values': True},
            LearningSet.dermatology: {'source': 'keel', 'filename': 'dermatology', 'missing_values': True},
            LearningSet.ecoli: {'source': 'keel', 'filename': 'ecoli', 'missing_values': False},
            LearningSet.flare: {'source': 'keel', 'filename': 'flare', 'missing_values': False},
            LearningSet.iris: {'source': 'keel', 'filename': 'iris', 'missing_values': False},
            LearningSet.kddcup: {'source': 'keel', 'filename': 'kddcup', 'missing_values': False},
            LearningSet.letter: {'source': 'keel', 'filename': 'letter', 'missing_values': False},
            LearningSet.mnist: {'source': 'mldata', 'filename': 'mnist', 'missing_values': False},
            LearningSet.poker: {'source': 'keel', 'filename': 'poker', 'missing_values': False},
            LearningSet.satimage: {'source': 'keel', 'filename': 'satimage', 'missing_values': False},
            LearningSet.segment: {'source': 'keel', 'filename': 'segment', 'missing_values': False},
            LearningSet.shuttle: {'source': 'keel', 'filename': 'shuttle', 'missing_values': False},
            LearningSet.vehicle: {'source': 'keel', 'filename': 'vehicle', 'missing_values': False},
            LearningSet.vowel: {'source': 'keel', 'filename': 'vowel', 'missing_values': False},
            LearningSet.wine: {'source': 'keel', 'filename': 'wine', 'missing_values': False},
            LearningSet.yeast: {'source': 'keel', 'filename': 'yeast', 'missing_values': False},
        }

        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)

    def get(self, learning_set: LearningSet) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        if not isinstance(learning_set, LearningSet):
            raise TypeError("Argument should be LearningSet enum type!")
        elif self.source_map[learning_set]['source'] == 'mldata':
            mldata_set = fetch_mldata(self.source_map[learning_set]['filename'], data_home=DATA_DIR)
            data = mldata_set.data.toarray()
            return data, mldata_set.target, data[0].tolist()
        elif self.source_map[learning_set]['source'] == 'keel':
            return self.keel_reader.read(**self.source_map[learning_set])

    def get_k_fold_generator(self, learning_set: LearningSet, cv: int):
        if not isinstance(learning_set, LearningSet):
            raise TypeError("Argument should be LearningSet enum type!")
        elif self.source_map[learning_set]['source'] == 'keel' and (cv == 10 or cv == 5):
            return self.keel_reader.make_k_fold_generator(**self.source_map[learning_set], cv=cv)
        else:
            return self.make_k_fold_generator(learning_set, cv)

    def make_k_fold_generator(self, learning_set: LearningSet, cv: int):
        path = os.path.join(DATA_DIR, self.source_map[learning_set]['source'])
        filename_beginning = "cv_" + str(cv) + "_" + str(learning_set)
        file_path = os.path.join(path, filename_beginning + ".pickle")

        if not self.check_cv_fold_exists(path, filename_beginning):
            self.save_folds(learning_set, cv, file_path)

        k_fold_splits = load(file_path)

        def k_fold_generator(learning_set: LearningSet, splits: List[Tuple[np.ndarray, np.ndarray]]):
            X, y, _ = self.get(learning_set)
            for train_index, test_index in splits:
                yield X[train_index], y[train_index], X[test_index], y[test_index]

        return k_fold_generator(learning_set, k_fold_splits)

    def check_cv_fold_exists(self, path: str, filename_beginning: str):
        dirs = os.listdir(path)
        for file in dirs:
            if file.startswith(filename_beginning):
                return True
        return False

    def save_folds(self, learning_set: LearningSet, cv: int, file_path:str):
        X, y, _ = self.get(learning_set)
        skf = StratifiedKFold(n_splits=cv)
        k_fold_split = skf.split(X, y)
        splits = []
        for train_index, test_index in k_fold_split:
            splits.append((train_index, test_index))
        dump(value=splits, filename=file_path, compress=3)
