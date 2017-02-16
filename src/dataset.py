from enum import Enum, unique
from typing import Tuple, List, Generator
import numpy as np
from src.readers.keel import KeelReader
from sklearn.datasets.mldata import fetch_mldata

from src.settings import DATA_DIR


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

    def get(self, learning_set: LearningSet) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        if not isinstance(learning_set, LearningSet):
            raise TypeError("Argument should be LearningSet enum type!")
        elif self.source_map[learning_set]['source'] == 'mldata':
            mldata_set = fetch_mldata(self.source_map[learning_set]['filename'], data_home=DATA_DIR)
            data = mldata_set.data.toarray()
            return data, mldata_set.target, data[0].tolist()
        elif self.source_map[learning_set]['source'] == 'keel':
            return self.keel_reader.read(**self.source_map[learning_set])

    def make_k_fold_generator(self, learning_set: LearningSet, cv: int):
        if not isinstance(learning_set, LearningSet):
            raise TypeError("Argument should be LearningSet enum type!")
        elif not self.source_map[learning_set]['source'] == 'keel':
            raise Exception("Function available for keel datasets")
        else:
            return self.keel_reader.make_k_fold_generator(**self.source_map[learning_set], cv=cv)
