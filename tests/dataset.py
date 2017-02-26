import unittest
from src.dataset import DataProvider, LearningSet


class DataProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = DataProvider()

    def test_should_validate_estimators_in_init(self):
        for parameter in [None, 'parametr']:
            with self.assertRaises(TypeError):
                self.provider.get(parameter)

    @unittest.skip
    def test_abalone(self):
        source_map = {
            LearningSet.abalone: {'instances': 4174, 'attributes': 10, 'classes': 28},
            LearningSet.breast: {'instances': 286, 'attributes': 43, 'classes': 2},
            LearningSet.cleveland: {'instances': 303, 'attributes': 13, 'classes': 5},
            LearningSet.dermatology: {'instances': 366, 'attributes': 34, 'classes': 6},
            LearningSet.ecoli: {'instances': 336, 'attributes': 7, 'classes': 8},
            LearningSet.flare: {'instances': 1066, 'attributes': 41, 'classes': 6},
            LearningSet.iris: {'instances': 150, 'attributes': 4, 'classes': 3},
            LearningSet.kddcup: {'instances': 494020, 'attributes': 142, 'classes': 23},
            LearningSet.letter: {'instances': 20000, 'attributes': 16, 'classes': 26},
            LearningSet.mnist: {'instances': 70000, 'attributes': 780, 'classes': 10},
            LearningSet.poker: {'instances': 1025009, 'attributes': 10, 'classes': 10},
            LearningSet.satimage: {'instances': 6435, 'attributes': 36, 'classes': 6},
            LearningSet.segment: {'instances': 2310, 'attributes': 19, 'classes': 7},
            LearningSet.shuttle: {'instances': 57999, 'attributes': 9, 'classes': 7},
            LearningSet.vehicle: {'instances': 846, 'attributes': 18, 'classes': 4},
            LearningSet.vowel: {'instances': 990, 'attributes': 13, 'classes': 11},
            LearningSet.wine: {'instances': 178, 'attributes': 13, 'classes': 3},
            LearningSet.yeast: {'instances': 1484, 'attributes': 8, 'classes': 10},
        }

        for learning_set in source_map:
            data, target, column_names = self.provider.get(learning_set)
            self.assertTupleEqual(data.shape, (source_map[learning_set]['instances'],
                                               source_map[learning_set]['attributes']))
            self.assertEqual(len(set(target)), source_map[learning_set]['classes'])
            self.assertEqual(len(column_names), source_map[learning_set]['attributes'])
