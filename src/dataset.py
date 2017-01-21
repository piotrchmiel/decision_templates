from enum import Enum, unique
from typing import Tuple

from sklearn.datasets.mldata import fetch_mldata

import numpy as np
from src.settings import DATA_DIR

@unique
class LearningSet(Enum):
    iris = 1


class DataProvider(object):

    def __init__(self):
        self.source_map = {
            LearningSet.iris: {'source': 'mldata', 'name': 'iris'}
        }

    def get(self, learning_set: LearningSet) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(learning_set, LearningSet):
            raise TypeError("Argument should be LearningSet enum type!")
        elif self.source_map[learning_set]['source'] == 'mldata':
            mldata_set = fetch_mldata(self.source_map[learning_set]['name'], data_home=DATA_DIR)
            return mldata_set.data, mldata_set.target
