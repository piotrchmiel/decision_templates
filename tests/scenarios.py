import unittest
from unittest.mock import Mock, patch
from copy import deepcopy
import src.decision_templates
from src.dataset import DataProvider, LearningSet
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from numpy import testing
import numpy as np


class ScenariosBase(object):
    def setUp(self):
        self.lr = LogisticRegression()
        self.knn = KNeighborsClassifier()
        self.gnb = GaussianNB()
        self.bnb = BernoulliNB()

        self.estimators = [('lr', self.lr), ('knn', self.knn), ('gnb', self.gnb), ('bnb', self.bnb)]
        self.provider = DataProvider()
        self.dectempl = src.decision_templates.DecisionTemplatesClassifier(estimators=self.estimators)


class ScenariosAverageTemplate(ScenariosBase, unittest.TestCase):
    def setUp(self):
        super(ScenariosAverageTemplate, self).setUp()
        self.target, self.labels = self.provider.get(LearningSet.iris)

        def side_effect_generator():
            support_result = np.array([0.0, 0.0, 0.0])
            while True:
                yield support_result
                support_result += [0.001, 0.01, 0.02]

        self.estimators_mock = deepcopy(self.estimators)

        for i, estimator in enumerate(self.estimators_mock):
            self.estimators_mock[i][1].predict_proba = Mock(side_effect=side_effect_generator())
            self.estimators_mock[i][1].classes_ = [0, 1, 2]

    def test_template_creation(self):
        for estimator_no in range(1, len(self.estimators) + 1):

            self.dectempl = src.decision_templates.DecisionTemplatesClassifier(
                estimators=self.estimators[:estimator_no])

            self.dectempl.fit(self.target, self.labels)
            testing.assert_array_equal(self.dectempl.classes_, [1, 2, 3])
            testing.assert_array_equal(self.dectempl.le_.transform(self.dectempl.classes_), [0, 1, 2])

            self.assertEqual(len(self.dectempl.templates_.keys()), 3)
            self.assertListEqual(list(self.dectempl.templates_.keys()), [0, 1, 2])

            for value in self.dectempl.templates_.values():
                self.assertTupleEqual(value.shape, (estimator_no, len(self.dectempl.classes_)))

    @patch('src.decision_templates.Parallel.__call__')
    def test_average_template_count(self, parallel):

        parallel.return_value = [estimator[1] for estimator in self.estimators_mock]

        self.dectempl = src.decision_templates.DecisionTemplatesClassifier(estimators=self.estimators_mock)
        self.dectempl.fit(self.target, self.labels)

        parallel.assert_called_once()

        testing.assert_array_equal(self.dectempl.classes_, [1, 2, 3])
        testing.assert_array_equal(self.dectempl.le_.transform(self.dectempl.classes_), [0, 1, 2])
        self.assertEqual(len(self.dectempl.templates_.keys()), 3)
        self.assertListEqual(list(self.dectempl.templates_.keys()), [0, 1, 2])

        for value in self.dectempl.templates_.values():
            self.assertTupleEqual(value.shape, (len(self.estimators), len(self.dectempl.classes_)))

        result_class_0 = np.asarray([[0.0245, 0.245, 0.49],
                                     [0.0245, 0.245, 0.49],
                                     [0.0245, 0.245, 0.49],
                                     [0.0245, 0.245, 0.49]])

        np.testing.assert_almost_equal(self.dectempl.templates_[0], result_class_0, 3)

        result_class_1 = np.asarray([[0.075,  0.745,  1.49],
                                    [0.075,  0.745,  1.49],
                                    [0.075,  0.745,  1.49],
                                    [0.075,  0.745,  1.49]])

        np.testing.assert_almost_equal(self.dectempl.templates_[1], result_class_1, 3)

        result_class_2 = np.asarray([[0.125,  1.245,  2.49],
                                     [0.125,  1.245,  2.49],
                                     [0.125,  1.245,  2.49],
                                     [0.125,  1.245,  2.49]])

        np.testing.assert_almost_equal(self.dectempl.templates_[2], result_class_2, 3)
