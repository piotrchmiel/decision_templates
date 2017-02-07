import unittest
from src.decision_templates import DecisionTemplatesClassifier
from sklearn.neighbors import KNeighborsClassifier


class ValidateParameters(unittest.TestCase):

    def test_should_validate_estimators_in_init(self):
        for parameter in [None, []]:
            with self.assertRaises(AttributeError):
                DecisionTemplatesClassifier(parameter)

    def test_should_validate_n_jobs_in_init(self):
        for parameter in [None, "10", "20", []]:
            with self.assertRaises(AttributeError):
                DecisionTemplatesClassifier(estimators=[("KNN", KNeighborsClassifier)], n_jobs=parameter)

    def test_should_validate_similarity_measure_in_init(self):
        for parameter in [None, "10", "20", []]:
            with self.assertRaises(ValueError):
                DecisionTemplatesClassifier(estimators=[("KNN", KNeighborsClassifier)], similarity_measure=parameter)

    def test_should_validate_template_creation_in_init(self):
        for parameter in [None, "10", "20", []]:
            with self.assertRaises(ValueError):
                DecisionTemplatesClassifier(estimators=[("KNN", KNeighborsClassifier)], template_creation=parameter)