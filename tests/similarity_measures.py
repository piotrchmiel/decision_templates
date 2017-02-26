import unittest
from unittest.mock import MagicMock
from src.decision_templates import DecisionTemplatesClassifier
from sklearn.neighbors import KNeighborsClassifier


class SimilarityMeasures(unittest.TestCase):
    def setUp(self):
        knn = KNeighborsClassifier()
        self.dectempl = DecisionTemplatesClassifier(estimators=[('knn', knn)])
        fake_list = MagicMock()
        fake_list.__len__.return_value = 2
        self.dectempl.estimators_ = fake_list
        self.dectempl.classes_ = fake_list

    def test_eucklidean_similarity_when_profile_equal_template(self):
        DT = [[0.89, 0.89], [0.89, 0.89]]
        DP = [[0.89, 0.89], [0.89, 0.89]]
        similarity = self.dectempl.euclidean_similarity(DT, DP)
        self.assertEqual(similarity, 1.0)

    def test_eucklidean_similarity_when_profile_not_equal_template(self):
        DT = [[0.89, 0.89], [0.89, 0.89]]
        DP = [[0.50, 0.30], [0.40, 0.10]]
        similarity = self.dectempl.euclidean_similarity(DT, DP)
        self.assertAlmostEqual(similarity, 0.6589)

        DT = [[0.89, 0.89], [0.89, 0.89]]
        DP = [[0.89, 0.30], [0.40, 0.10]]
        similarity = self.dectempl.euclidean_similarity(DT, DP)
        self.assertAlmostEqual(similarity, 0.6969, places=4)
