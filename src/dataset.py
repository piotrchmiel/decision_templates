import os
from typing import Tuple, List

import numpy as np
from sklearn.datasets.mldata import fetch_mldata

import src.readers.keel
import src.utils.k_fold_creator
from src.learning_set import LearningSet
from src.settings import DATA_DIR


class DataProvider(object):

    def __init__(self):
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
            return src.readers.keel.KeelReader(learning_set, **self.source_map[learning_set]).make_dataset()

    def get_k_fold_generator(self, learning_set: LearningSet, cv: int):
        if not isinstance(learning_set, LearningSet):
            raise TypeError("Argument should be LearningSet enum type!")
        elif self.source_map[learning_set]['source'] == 'keel' and (cv == 10 or cv == 5):
            return src.readers.keel.KeelReader(learning_set, **self.source_map[learning_set]).make_k_fold_generator(cv)
        else:
            X, y, _ = self.get(learning_set)
            return src.utils.k_fold_creator.DefaultKFoldCreator(
                learning_set, X, y, self.source_map[learning_set]['source']).make_k_fold_generator(cv)