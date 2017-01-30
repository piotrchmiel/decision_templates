import unittest
from unittest.mock import MagicMock, Mock
from src.decision_templates import DecisionTemplatesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


class SimilarityMeasures(unittest.TestCase):
    def setUp(self):
        lr = LogisticRegression()
        knn = KNeighborsClassifier()
        gnb = GaussianNB()
        self.dectempl = DecisionTemplatesClassifier(estimators=[('lr', lr), ('knn', knn), ('gnb', gnb)])

    def test_eucklidean_similarity_when_profile_equal_template(self):
        fake_list = MagicMock()
        fake_list.__len__.return_value = 2
        self.dectempl.estimators_ = fake_list
        self.dectempl.classes_ = fake_list

        DT = [[0.89, 0.89], [0.89, 0.89]]
        DP = [[0.89, 0.89], [0.89, 0.89]]
        similarity = self.dectempl.eucklidean_similarity(DT, DP)
        self.assertEqual(similarity, 1.0)

    def test_eucklidean_similarity_when_profile_not_equal_template(self):

        fake_list = MagicMock()
        fake_list.__len__.return_value = 2
        self.dectempl.estimators_ = fake_list
        self.dectempl.classes_ = fake_list

        DT = [[0.89, 0.89], [0.89, 0.89]]
        DP = [[0.50, 0.30], [0.40, 0.10]]
        similarity = self.dectempl.eucklidean_similarity(DT, DP)
        self.assertAlmostEqual(similarity, 0.6589)

        DT = [[0.89, 0.89], [0.89, 0.89]]
        DP = [[0.89, 0.30], [0.40, 0.10]]
        similarity = self.dectempl.eucklidean_similarity(DT, DP)
        self.assertAlmostEqual(similarity, 0.6969, places=4)
