import unittest
from src.decision_templates import DecisionTemplatesClassifier
from sklearn.neighbors import KNeighborsClassifier


class ValidateParameters(unittest.TestCase):

    def test_should_validate_estimators_in_init(self):
        for parameter in [None, []]:
            with self.assertRaises(AttributeError):
                DecisionTemplatesClassifier(parameter)

    def test_should_validate_groups_mapping_in_init(self):
        for parameter in [[], [1], [tuple([])]]:
            with self.assertRaises(AttributeError):
                DecisionTemplatesClassifier(estimators=[("KNN", KNeighborsClassifier)], groups_mapping=parameter)

        with self.assertRaises(AttributeError):
            DecisionTemplatesClassifier(estimators=[("KNN", KNeighborsClassifier),
                                                    ("KNN", KNeighborsClassifier)], groups_mapping=[(1,)])

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
                DecisionTemplatesClassifier(estimators=[("KNN", KNeighborsClassifier)], template_construction=parameter)

    def test_should_validate_decision_similarity_in_init(self):
        for parameter in [None, "10", "20", []]:
            with self.assertRaises(ValueError):
                DecisionTemplatesClassifier(estimators=[("KNN", KNeighborsClassifier)], similarity_for_group=parameter)

    def test_should_validate_decision_strategy_in_init(self):
        for parameter in [None, "10", "20", []]:
            with self.assertRaises(ValueError):
                DecisionTemplatesClassifier(estimators=[("KNN", KNeighborsClassifier)], decision_strategy=parameter)

    def test_should_validate_k_similar_templates_in_init(self):
        for parameter in [None, -10, 0, []]:
            with self.assertRaises(AttributeError):
                DecisionTemplatesClassifier(estimators=[("KNN", KNeighborsClassifier)], k_similar_templates=parameter)