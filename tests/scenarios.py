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


    def test_create_fit_one_template_per_class_by_avg_classical_kuncheva_algorithm(self):
        for estimator_no in range(1, len(self.estimators) + 1):

            self.dectempl = src.decision_templates.DecisionTemplatesClassifier(
                estimators=self.estimators[:estimator_no])

            self.dectempl.fit(self.target, self.labels)
            testing.assert_array_equal(self.dectempl.classes_, [1, 2, 3])
            testing.assert_array_equal(self.dectempl.le_.transform(self.dectempl.classes_), [0, 1, 2])

            self.assertEqual(len(self.dectempl.templates_.keys()), 3)
            self.assertListEqual(list(self.dectempl.templates_.keys()), [0, 1, 2])

            for class_templates in self.dectempl.templates_.values():
                self.assertEqual(len(class_templates), 1)
                self.assertTupleEqual(class_templates[0].shape, (estimator_no, len(self.dectempl.classes_)))

    def test_predict_one_template_per_class_by_avg_classical_kuncheva_algorithm(self):
        self.dectempl.fit(self.target, self.labels)
        np.testing.assert_array_equal(self.dectempl.predict_proba([self.target[0]]),
                         [[0.99913655057051631, 0.52665233786702026, 0.4507965644527222]])
        self.assertEqual(self.dectempl.predict([self.target[0]]), 1)
        np.testing.assert_array_equal(self.dectempl.predict_proba([self.target[1]]),
                                      [[0.99631684202062598, 0.5843003218440872, 0.50063006479511785]])
        np.testing.assert_array_equal(self.dectempl.predict_proba(self.target[0:2]),
                                      [[0.99913655057051631, 0.52665233786702026, 0.4507965644527222],
                                       [0.99631684202062598, 0.5843003218440872, 0.50063006479511785]])
        np.testing.assert_array_equal(self.dectempl.predict(self.target[0:2]), [1, 1])


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

        for class_templates in self.dectempl.templates_.values():
            self.assertEqual(len(class_templates), 1)
            self.assertTupleEqual(class_templates[0].shape, (len(self.estimators), len(self.dectempl.classes_)))

        result_class_0 = np.asarray([[0.0245, 0.245, 0.49],
                                     [0.0245, 0.245, 0.49],
                                     [0.0245, 0.245, 0.49],
                                     [0.0245, 0.245, 0.49]])

        np.testing.assert_almost_equal(self.dectempl.templates_[0][0], result_class_0, 3)

        result_class_1 = np.asarray([[0.075,  0.745,  1.49],
                                    [0.075,  0.745,  1.49],
                                    [0.075,  0.745,  1.49],
                                    [0.075,  0.745,  1.49]])

        np.testing.assert_almost_equal(self.dectempl.templates_[1][0], result_class_1, 3)

        result_class_2 = np.asarray([[0.125,  1.245,  2.49],
                                     [0.125,  1.245,  2.49],
                                     [0.125,  1.245,  2.49],
                                     [0.125,  1.245,  2.49]])

        np.testing.assert_almost_equal(self.dectempl.templates_[2][0], result_class_2, 3)

    def test_fit_one_template_for_each_class_by_avg(self):
        fake_y = ['red', 'red', 'green', 'blue', 'green', 'blue', 'green']
        X = self.target[:7]
        curr_sample_weight = np.ones((len(fake_y)))

        self.dectempl = src.decision_templates.DecisionTemplatesClassifier(estimators=self.estimators_mock)
        self.dectempl.estimators_ = [estimator[1] for estimator in self.estimators_mock]
        self.dectempl.le_.transform = Mock(return_value=[0, 1, 2])
        self.dectempl.classes_ = ['red', 'green', 'blue']

        result = self.dectempl._fit_one_template_for_each_class_by_avg(X, fake_y, curr_sample_weight)

        for class_template in result.values():
            self.assertTupleEqual(class_template.shape, (4, 3))

        result_class_red = np.asarray([[0.0005,  0.005,  0.01],
                                       [0.0005,  0.005,  0.01],
                                       [0.0005,  0.005,  0.01],
                                       [0.0005,  0.005,  0.01]])

        np.testing.assert_array_equal(result['red'], result_class_red)

        result_class_green = np.asarray([[0.004,  0.04,  0.08],
                                         [0.004,  0.04,  0.08],
                                         [0.004,  0.04,  0.08],
                                         [0.004,  0.04,  0.08]])

        np.testing.assert_array_equal(result['green'], result_class_green)

        result_class_blue= np.asarray([[0.004,  0.04,  0.08],
                                       [0.004,  0.04,  0.08],
                                       [0.004,  0.04,  0.08],
                                       [0.004,  0.04,  0.08]])

        np.testing.assert_array_equal(result['blue'], result_class_blue)

    def test_fit_one_template_for_each_class_by_avg_with_weights(self):
        fake_y = ['red', 'red', 'green', 'blue', 'green', 'blue', 'green']
        X = self.target[:7]
        curr_sample_weight = [0.0, 0.0, 2.0, 1.0, 1.0, 1.0, 3.0]

        self.dectempl = src.decision_templates.DecisionTemplatesClassifier(estimators=self.estimators_mock)
        self.dectempl.estimators_ = [estimator[1] for estimator in self.estimators_mock]
        self.dectempl.le_.transform = Mock(return_value=[0, 1, 2])
        self.dectempl.classes_ = ['red', 'green', 'blue']

        result = self.dectempl._fit_one_template_for_each_class_by_avg(X, fake_y, curr_sample_weight)

        for class_template in result.values():
            self.assertTupleEqual(class_template.shape, (4, 3))

        result_class_red = np.asarray([[0.0,  0.0,  0.0],
                                       [0.0,  0.0,  0.0],
                                       [0.0,  0.0,  0.0],
                                       [0.0,  0.0,  0.0]])

        np.testing.assert_array_equal(result['red'], result_class_red)

        result_class_green = np.asarray([[0.004333,  0.043333,  0.086667],
                                         [0.004333,  0.043333,  0.086667],
                                         [0.004333,  0.043333,  0.086667],
                                         [0.004333,  0.043333,  0.086667]])

        np.testing.assert_array_almost_equal(result['green'], result_class_green)

        result_class_blue = np.asarray([[0.004,  0.04,  0.08],
                                       [0.004,  0.04,  0.08],
                                       [0.004,  0.04,  0.08],
                                       [0.004,  0.04,  0.08]])

        np.testing.assert_array_equal(result['blue'], result_class_blue)

    def test_fit_one_template_for_each_class_by_med(self):
        fake_y = ['red', 'red', 'green', 'blue', 'green', 'blue', 'green']
        X = self.target[:7]
        curr_sample_weight = np.ones((len(fake_y)))

        self.dectempl = src.decision_templates.DecisionTemplatesClassifier(estimators=self.estimators_mock)
        self.dectempl.estimators_ = [estimator[1] for estimator in self.estimators_mock]
        self.dectempl.le_.transform = Mock(return_value=[0, 1, 2])
        self.dectempl.classes_ = ['red', 'green', 'blue']

        result = self.dectempl._fit_one_template_for_each_class_by_med(X, fake_y, curr_sample_weight)

        for class_template in result.values():
            self.assertTupleEqual(class_template.shape, (4, 3))

        result_class_red = np.asarray([[0.0005, 0.005, 0.01],
                                       [0.0005, 0.005, 0.01],
                                       [0.0005, 0.005, 0.01],
                                       [0.0005, 0.005, 0.01]])

        np.testing.assert_array_equal(result['red'], result_class_red)

        result_class_green = np.asarray([[0.004, 0.04, 0.08],
                                         [0.004, 0.04, 0.08],
                                         [0.004, 0.04, 0.08],
                                         [0.004, 0.04, 0.08]])

        np.testing.assert_array_equal(result['green'], result_class_green)

        result_class_blue = np.asarray([[0.004, 0.04, 0.08],
                                        [0.004, 0.04, 0.08],
                                        [0.004, 0.04, 0.08],
                                        [0.004, 0.04, 0.08]])

        np.testing.assert_array_equal(result['blue'], result_class_blue)

    def test_fit_one_template_for_each_class_by_med_with_weights(self):
        fake_y = ['red', 'red', 'green', 'blue', 'green', 'blue', 'green']
        X = self.target[:7]
        curr_sample_weight = [0.0, 2.0, 2.0, 1.0, 1.0, 1.0, 3.0]

        self.dectempl = src.decision_templates.DecisionTemplatesClassifier(estimators=self.estimators_mock)
        self.dectempl.estimators_ = [estimator[1] for estimator in self.estimators_mock]
        self.dectempl.le_.transform = Mock(return_value=[0, 1, 2])
        self.dectempl.classes_ = ['red', 'green', 'blue']

        result = self.dectempl._fit_one_template_for_each_class_by_med(X, fake_y, curr_sample_weight)

        for class_template in result.values():
            self.assertTupleEqual(class_template.shape, (4, 3))

        result_class_red = np.asarray([[0.001, 0.01, 0.02],
                                       [0.001, 0.01, 0.02],
                                       [0.001, 0.01, 0.02],
                                       [0.001, 0.01, 0.02]])

        np.testing.assert_array_equal(result['red'], result_class_red)

        result_class_green = np.asarray([[0.005,  0.05,  0.1],
                                         [0.005,  0.05,  0.1],
                                         [0.005,  0.05,  0.1],
                                         [0.005,  0.05,  0.1]])

        np.testing.assert_array_almost_equal(result['green'], result_class_green)

        result_class_blue = np.asarray([[0.004,  0.04,  0.08],
                                        [0.004,  0.04,  0.08],
                                        [0.004,  0.04,  0.08],
                                        [0.004,  0.04,  0.08]])

        np.testing.assert_array_equal(result['blue'], result_class_blue)