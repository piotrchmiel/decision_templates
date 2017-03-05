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


class Scenarios(unittest.TestCase):
    def setUp(self):
        self.lr = LogisticRegression()
        self.knn = KNeighborsClassifier()
        self.gnb = GaussianNB()
        self.bnb = BernoulliNB()

        self.estimators = [('lr', self.lr), ('knn', self.knn), ('gnb', self.gnb), ('bnb', self.bnb)]
        self.provider = DataProvider()
        self.dectempl = src.decision_templates.DecisionTemplatesClassifier(estimators=self.estimators)
        self.target, self.labels, _ = self.provider.get(LearningSet.iris)

        def side_effect_generator():
            support_result = np.array([0.0, 0.0, 0.0])
            while True:
                yield support_result
                support_result += [0.001, 0.01, 0.02]

        self.estimators_mock = deepcopy(self.estimators)

        for i in range(len(self.estimators_mock)):
            self.estimators_mock[i][1].predict_proba = Mock(side_effect=side_effect_generator())
            self.estimators_mock[i][1].classes_ = [0, 1, 2]

    def test_create_fit_one_template_per_class_one_group(self):
        group = 1
        for estimator_no in range(1, len(self.estimators) + 1):
            self.dectempl = src.decision_templates.DecisionTemplatesClassifier(
                estimators=self.estimators[:estimator_no])

            self.dectempl.fit(self.target, self.labels)
            testing.assert_array_equal(self.dectempl.classes_, ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
            testing.assert_array_equal(self.dectempl.le_.transform(self.dectempl.classes_), [0, 1, 2])

            self.assertEqual(len(self.dectempl.templates_.keys()), 3)
            self.assertListEqual(list(self.dectempl.templates_.keys()), [0, 1, 2])
            self.assertEqual(len(self.dectempl.groups_.keys()), 1)

            for class_templates in self.dectempl.templates_.values():
                self.assertEqual(len(class_templates[group]), 1)
                self.assertTupleEqual(class_templates[group][0].shape, (estimator_no, len(self.dectempl.classes_)))

    def test_create_fit_ten_templates_per_class_one_group(self):
        group = 1
        for estimator_no in range(1, len(self.estimators) + 1):
            self.dectempl = src.decision_templates.DecisionTemplatesClassifier(
                estimators=self.estimators[:estimator_no], template_fit_strategy='random_subspace', n_templates=10)

            self.dectempl.fit(self.target, self.labels)
            testing.assert_array_equal(self.dectempl.classes_, ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
            testing.assert_array_equal(self.dectempl.le_.transform(self.dectempl.classes_), [0, 1, 2])

            self.assertEqual(len(self.dectempl.templates_.keys()), 3)
            self.assertListEqual(list(self.dectempl.templates_.keys()), [0, 1, 2])
            self.assertEqual(len(self.dectempl.groups_.keys()), 1)

            for class_templates in self.dectempl.templates_.values():
                self.assertEqual(len(class_templates[group]), 10)
                for template in class_templates[group]:
                    self.assertTupleEqual(template.shape, (estimator_no, len(self.dectempl.classes_)))

    def test_create_fit_one_template_per_class_three_groups(self):
        self.dectempl = src.decision_templates.DecisionTemplatesClassifier(estimators=self.estimators,
                                                                           groups_mapping=[
                                                                               (1, 2), (1, 2, 3), (1, 3), (1, 2)])
        self.dectempl.fit(self.target, self.labels)
        testing.assert_array_equal(self.dectempl.classes_, ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
        testing.assert_array_equal(self.dectempl.le_.transform(self.dectempl.classes_), [0, 1, 2])
        self.assertEqual(len(self.dectempl.templates_.keys()), 3)
        self.assertListEqual(list(self.dectempl.templates_.keys()), [0, 1, 2])
        self.assertEqual(len(self.dectempl.groups_.keys()), 3)

        for class_templates in self.dectempl.templates_.values():
            for group in self.dectempl.groups_.keys():
                self.assertEqual(len(class_templates[group]), 1)
                self.assertTupleEqual(class_templates[group][0].shape, (self.dectempl.groups_[group],
                                                                        len(self.dectempl.classes_)))

    def test_create_fit_five_templates_per_class_three_groups(self):
        self.dectempl = src.decision_templates.DecisionTemplatesClassifier(estimators=self.estimators,
                                                                           groups_mapping=[
                                                                               (1, 2), (1, 2, 3), (1, 3), (1, 2)],
                                                                           template_fit_strategy='bootstrap',
                                                                           n_templates=5)
        self.dectempl.fit(self.target, self.labels)
        testing.assert_array_equal(self.dectempl.classes_, ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
        testing.assert_array_equal(self.dectempl.le_.transform(self.dectempl.classes_), [0, 1, 2])
        self.assertEqual(len(self.dectempl.templates_.keys()), 3)
        self.assertListEqual(list(self.dectempl.templates_.keys()), [0, 1, 2])
        self.assertEqual(len(self.dectempl.groups_.keys()), 3)

        for class_templates in self.dectempl.templates_.values():
            for group in self.dectempl.groups_.keys():
                self.assertEqual(len(class_templates[group]), 5)
                for template in class_templates[group]:
                    self.assertTupleEqual(template.shape, (self.dectempl.groups_[group], len(self.dectempl.classes_)))

    def test_predict_one_template_per_class_by_avg_classical_kuncheva_algorithm(self):
        self.dectempl.fit(self.target, self.labels)
        np.testing.assert_array_almost_equal(self.dectempl.predict_proba([self.target[0]]),
                                      [[0.99996380550968655, 0.59543293656855445, 0.58100665521394679]])
        self.assertEqual(self.dectempl.predict([self.target[0]]), 'Iris-setosa')
        np.testing.assert_array_almost_equal(self.dectempl.predict_proba([self.target[1]]),
                                      [[0.99929086373833642, 0.6131342371589803, 0.59414955137607195]])
        self.assertEqual(self.dectempl.predict([self.target[1]]), 'Iris-setosa')
        np.testing.assert_array_almost_equal(self.dectempl.predict_proba(self.target[0:2]),
                                      [[0.99996380550968655, 0.59543293656855445, 0.58100665521394679],
                                       [0.99929086373833642, 0.6131342371589803, 0.59414955137607195]])
        np.testing.assert_array_equal(self.dectempl.predict(self.target[0:2]), ['Iris-setosa', 'Iris-setosa'])

    def test_fit_one_template_for_each_class_by_avg(self):
        group = 1
        fake_y = ['red', 'red', 'green', 'blue', 'green', 'blue', 'green']
        X = self.target[:7]
        curr_sample_weight = np.ones((len(fake_y)))

        self.dectempl = src.decision_templates.DecisionTemplatesClassifier(estimators=self.estimators_mock)
        self.dectempl.estimators_ = [estimator[1] for estimator in self.estimators_mock]
        self.dectempl.le_.transform = Mock(return_value=[0, 1, 2])
        self.dectempl.classes_ = ['red', 'green', 'blue']

        result = self.dectempl._fit_one_template_for_each_class_by_avg(group, X, fake_y, curr_sample_weight)

        for class_template in result.values():
            self.assertTupleEqual(class_template.shape, (4, 3))

        result_class_red = np.asarray([[0.0005,  0.005,  0.01],
                                       [0.0005,  0.005,  0.01],
                                       [0.0005,  0.005,  0.01],
                                       [0.0005,  0.005,  0.01]])

        np.testing.assert_array_almost_equal(result['red'], result_class_red)

        result_class_green = np.asarray([[0.004,  0.04,  0.08],
                                         [0.004,  0.04,  0.08],
                                         [0.004,  0.04,  0.08],
                                         [0.004,  0.04,  0.08]])

        np.testing.assert_array_almost_equal(result['green'], result_class_green)

        result_class_blue= np.asarray([[0.004,  0.04,  0.08],
                                       [0.004,  0.04,  0.08],
                                       [0.004,  0.04,  0.08],
                                       [0.004,  0.04,  0.08]])

        np.testing.assert_array_almost_equal(result['blue'], result_class_blue)


    def test_fit_one_template_for_each_class_by_avg_with_weights(self):
        group = 1
        fake_y = ['red', 'red', 'green', 'blue', 'green', 'blue', 'green']
        X = self.target[:7]
        curr_sample_weight = [0.0, 0.0, 2.0, 1.0, 1.0, 1.0, 3.0]

        self.dectempl = src.decision_templates.DecisionTemplatesClassifier(estimators=self.estimators_mock)
        self.dectempl.estimators_ = [estimator[1] for estimator in self.estimators_mock]
        self.dectempl.le_.transform = Mock(return_value=[0, 1, 2])
        self.dectempl.classes_ = ['red', 'green', 'blue']

        result = self.dectempl._fit_one_template_for_each_class_by_avg(group, X, fake_y, curr_sample_weight)

        for class_template in result.values():
            self.assertTupleEqual(class_template.shape, (4, 3))

        result_class_red = np.asarray([[0.0,  0.0,  0.0],
                                       [0.0,  0.0,  0.0],
                                       [0.0,  0.0,  0.0],
                                       [0.0,  0.0,  0.0]])

        np.testing.assert_array_almost_equal(result['red'], result_class_red)

        result_class_green = np.asarray([[0.004333,  0.043333,  0.086667],
                                         [0.004333,  0.043333,  0.086667],
                                         [0.004333,  0.043333,  0.086667],
                                         [0.004333,  0.043333,  0.086667]])

        np.testing.assert_array_almost_equal(result['green'], result_class_green)

        result_class_blue = np.asarray([[0.004,  0.04,  0.08],
                                       [0.004,  0.04,  0.08],
                                       [0.004,  0.04,  0.08],
                                       [0.004,  0.04,  0.08]])

        np.testing.assert_array_almost_equal(result['blue'], result_class_blue)

    def test_fit_one_template_for_each_class_by_med(self):
        group = 1
        fake_y = ['red', 'red', 'green', 'blue', 'green', 'blue', 'green']
        X = self.target[:7]
        curr_sample_weight = np.ones((len(fake_y)))

        self.dectempl = src.decision_templates.DecisionTemplatesClassifier(estimators=self.estimators_mock)
        self.dectempl.estimators_ = [estimator[1] for estimator in self.estimators_mock]
        self.dectempl.le_.transform = Mock(return_value=[0, 1, 2])
        self.dectempl.classes_ = ['red', 'green', 'blue']

        result = self.dectempl._fit_one_template_for_each_class_by_med(group, X, fake_y, curr_sample_weight)

        for class_template in result.values():
            self.assertTupleEqual(class_template.shape, (4, 3))

        result_class_red = np.asarray([[0.0005, 0.005, 0.01],
                                       [0.0005, 0.005, 0.01],
                                       [0.0005, 0.005, 0.01],
                                       [0.0005, 0.005, 0.01]])

        np.testing.assert_array_almost_equal(result['red'], result_class_red)

        result_class_green = np.asarray([[0.004, 0.04, 0.08],
                                         [0.004, 0.04, 0.08],
                                         [0.004, 0.04, 0.08],
                                         [0.004, 0.04, 0.08]])

        np.testing.assert_array_almost_equal(result['green'], result_class_green)

        result_class_blue = np.asarray([[0.004, 0.04, 0.08],
                                        [0.004, 0.04, 0.08],
                                        [0.004, 0.04, 0.08],
                                        [0.004, 0.04, 0.08]])

        np.testing.assert_array_almost_equal(result['blue'], result_class_blue)

    def test_fit_one_template_for_each_class_by_med_with_weights(self):
        group = 1
        fake_y = ['red', 'red', 'green', 'blue', 'green', 'blue', 'green']
        X = self.target[:7]
        curr_sample_weight = [0.0, 2.0, 2.0, 1.0, 1.0, 1.0, 3.0]

        self.dectempl = src.decision_templates.DecisionTemplatesClassifier(estimators=self.estimators_mock)
        self.dectempl.estimators_ = [estimator[1] for estimator in self.estimators_mock]
        self.dectempl.le_.transform = Mock(return_value=[0, 1, 2])
        self.dectempl.classes_ = ['red', 'green', 'blue']

        result = self.dectempl._fit_one_template_for_each_class_by_med(group, X, fake_y, curr_sample_weight)

        for class_template in result.values():
            self.assertTupleEqual(class_template.shape, (4, 3))

        result_class_red = np.asarray([[0.001, 0.01, 0.02],
                                       [0.001, 0.01, 0.02],
                                       [0.001, 0.01, 0.02],
                                       [0.001, 0.01, 0.02]])

        np.testing.assert_array_almost_equal(result['red'], result_class_red)

        result_class_green = np.asarray([[0.005,  0.05,  0.1],
                                         [0.005,  0.05,  0.1],
                                         [0.005,  0.05,  0.1],
                                         [0.005,  0.05,  0.1]])

        np.testing.assert_array_almost_equal(result['green'], result_class_green)

        result_class_blue = np.asarray([[0.004,  0.04,  0.08],
                                        [0.004,  0.04,  0.08],
                                        [0.004,  0.04,  0.08],
                                        [0.004,  0.04,  0.08]])

        np.testing.assert_array_almost_equal(result['blue'], result_class_blue)