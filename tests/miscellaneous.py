import unittest
from unittest.mock import Mock
from src.decision_templates import DecisionTemplatesClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class MiscellaneousTests(unittest.TestCase):
    def setUp(self):
        knn = KNeighborsClassifier()
        knn.predict_proba = Mock(return_value=[0.455, 0.455])
        knn.classes_ = [1, 2]
        self.dectempl = DecisionTemplatesClassifier(estimators=[('knn', knn), ('knn1', knn)])
        self.dectempl.estimators_ = [knn, knn]
        self.dectempl.classes_ = [1, 2]
        self.dectempl.le_.transform = Mock(return_value=[1, 2])

    def test_decision_profile_creation(self):
        decision_profile = self.dectempl._make_decision_profile(group=1, x=np.asarray([0.4334, 0.4344, 4334, 4343],
                                                                                      dtype=np.float64))
        np.testing.assert_array_equal([[0.455, 0.455], [0.455, 0.455]], decision_profile)