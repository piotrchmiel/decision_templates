import os
import warnings
from src.dataset import DataProvider, LearningSet
from src.decision_templates import DecisionTemplatesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.externals.joblib import Parallel, delayed, dump, load
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection._validation import _score
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from typing import List
from copy import deepcopy
from src.writers.excel import XlsxWriter
import numpy as np
from src.settings import PICKLED_ESTIMATORS_DIR, RESULTS_SHEET


class Benchmark(object):
    def __init__(self):
        self.provider = DataProvider()
        if not os.path.exists(PICKLED_ESTIMATORS_DIR):
            os.mkdir(PICKLED_ESTIMATORS_DIR)

    def cross_validation_score(self, estimator: DecisionTemplatesClassifier, learningSet: LearningSet,
                               groups: np.ndarray = None, scoring: str = None, cv: int = 10, n_jobs: int = 1,
                               verbose: int = 0, fit_params: dict = None) -> List[float]:
        try:
            keel_iterator = self.provider.make_k_fold_generator(learningSet, cv)
        except:
            X, y, _ = self.provider.get(learningSet)
            return cross_val_score(estimator, X, y, groups, scoring, cv, n_jobs, verbose, fit_params)
        else:
            scorer = check_scoring(estimator, scoring=None)
            parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
            scores = parallel(delayed(self._fit_and_score)(deepcopy(estimator), scorer, train_X, train_y, test_X, test_y)
                          for train_X, train_y, test_X, test_y in keel_iterator)
            return scores

    def _fit_and_score(self, esitmator: DecisionTemplatesClassifier, scorer, train_X, train_y, test_X, test_y):
        esitmator.fit(train_X, train_y)
        return _score(esitmator, test_X, test_y, scorer)

    def cross_validation_score_10_folds(self, parameters: dict, learningSet: LearningSet, n_jobs: int = 1,
                               verbose: int = 0) -> List[float]:

        keel_iterator = self.provider.make_k_fold_generator(learningSet, 10)
        parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
        scores = parallel(delayed(self._fit_and_score_10_folds)(
            deepcopy(parameters), fold, learningSet, train_X, train_y, test_X, test_y)
            for fold, (train_X, train_y, test_X, test_y) in enumerate(keel_iterator))

        return scores

    def _fit_and_score_10_folds(self, parameters, fold, learningSet, train_X, train_y, test_X, test_y):
        estimators = self.load_base_estimators(learningSet, fold)
        parameters['estimators'] = estimators
        estimator = DecisionTemplatesClassifier(**parameters)
        estimator.fit(train_X, train_y)
        scorer = check_scoring(estimator, scoring=None)
        return _score(estimator, test_X, test_y, scorer)

    def fit_and_save_base_estimators(self, estimators, learningSet: LearningSet, cv: int = 10):
        keel_iterator = self.provider.make_k_fold_generator(learningSet, cv)

        parallel = Parallel(n_jobs=1, verbose=0)
        parallel(delayed(self.fit_base_estimators)(fold, learningSet, deepcopy(estimators), train_X, train_y)
                                              for fold, (train_X, train_y, _, _) in enumerate(keel_iterator))

    def fit_base_estimators(self, fold, learning_set, estimator_tup, X, y):
        le_ = LabelEncoder()
        le_.fit(y)
        transformed_y = le_.transform(y)
        for named_estimator in estimator_tup:
            named_estimator[1].fit(X, transformed_y)

        path = os.path.join(PICKLED_ESTIMATORS_DIR, "fold_" + str(fold + 1) + "_" + str(learning_set) + ".pickle")
        dump(estimator_tup, path, compress=4)

    def load_base_estimators(self, learning_set, fold):
        path = os.path.join(PICKLED_ESTIMATORS_DIR, "fold_" + str(fold + 1) + "_" + str(learning_set) + ".pickle")
        return load(path)

    testing_variants = {'algorithm_1':
                            {'estimators': None, 'groups_mapping': None, 'similarity_measure': 'euclidean',
                             'template_construction': 'avg', 'template_fit_strategy': 'one_per_class',
                             'n_templates': 1, 'decision_strategy': 'most_similar_template',
                             'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                        'algorithm_2':
                            {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                             'similarity_measure': 'euclidean',
                             'template_construction': 'avg', 'template_fit_strategy': 'one_per_class',
                             'n_templates': 1, 'decision_strategy': 'most_similar_template',
                             'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                        'algorithm_3':
                            {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                             'similarity_measure': 'euclidean',
                             'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                             'n_templates': 5, 'decision_strategy': 'most_similar_template',
                             'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                        'algorithm_4':
                            {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                             'similarity_measure': 'euclidean',
                             'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                             'n_templates': 10, 'decision_strategy': 'most_similar_template',
                             'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                        'algorithm_5':
                            {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                             'similarity_measure': 'euclidean',
                             'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                             'n_templates': 15, 'decision_strategy': 'most_similar_template',
                             'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                        'algorithm_6':
                            {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                             'similarity_measure': 'euclidean',
                             'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                             'n_templates': 5, 'decision_strategy': 'most_similar_template',
                             'similarity_for_group': 'average_group', 'k_similar_templates': 1, 'n_jobs': -1},
                        'algorithm_7':
                            {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                             'similarity_measure': 'euclidean',
                             'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                             'n_templates': 10, 'decision_strategy': 'most_similar_template',
                             'similarity_for_group': 'average_group', 'k_similar_templates': 1, 'n_jobs': -1},
                        'algorithm_8':
                            {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                             'similarity_measure': 'euclidean',
                             'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                             'n_templates': 15, 'decision_strategy': 'most_similar_template',
                             'similarity_for_group': 'average_group', 'k_similar_templates': 1, 'n_jobs': -1},
                        'algorithm_9':
                            {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                             'similarity_measure': 'euclidean',
                             'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                             'n_templates': 5, 'decision_strategy': 'most_similar_template',
                             'similarity_for_group': 'sum_group', 'k_similar_templates': 1, 'n_jobs': -1},
                        'algorithm_10':
                            {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                             'similarity_measure': 'euclidean',
                             'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                             'n_templates': 10, 'decision_strategy': 'most_similar_template',
                             'similarity_for_group': 'sum_group', 'k_similar_templates': 1, 'n_jobs': -1},
                        'algorithm_11':
                            {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                             'similarity_measure': 'euclidean',
                             'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                             'n_templates': 15, 'decision_strategy': 'most_similar_template',
                             'similarity_for_group': 'sum_group', 'k_similar_templates': 1, 'n_jobs': -1},
                        'algorithm_12':
                            {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                             'similarity_measure': 'euclidean',
                             'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                             'n_templates': 5, 'decision_strategy': 'most_similar_template',
                             'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                        'algorithm_13':
                            {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                             'similarity_measure': 'euclidean',
                             'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                             'n_templates': 10, 'decision_strategy': 'most_similar_template',
                             'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                        'algorithm_14':
                            {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                             'similarity_measure': 'euclidean',
                             'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                             'n_templates': 15, 'decision_strategy': 'most_similar_template',
                             'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                        'algorithm_15':
                            {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                             'similarity_measure': 'euclidean',
                             'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                             'n_templates': 5, 'decision_strategy': 'most_similar_template',
                             'similarity_for_group': 'average_group', 'k_similar_templates': 1, 'n_jobs': -1},
                        'algorithm_16':
                            {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                             'similarity_measure': 'euclidean',
                             'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                             'n_templates': 10, 'decision_strategy': 'most_similar_template',
                             'similarity_for_group': 'average_group', 'k_similar_templates': 1, 'n_jobs': -1},
                        'algorithm_17':
                            {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                             'similarity_measure': 'euclidean',
                             'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                             'n_templates': 15, 'decision_strategy': 'most_similar_template',
                             'similarity_for_group': 'average_group', 'k_similar_templates': 1, 'n_jobs': -1},
                        'algorithm_18':
                            {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                             'similarity_measure': 'euclidean',
                             'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                             'n_templates': 5, 'decision_strategy': 'most_similar_template',
                             'similarity_for_group': 'sum_group', 'k_similar_templates': 1, 'n_jobs': -1},
                        'algorithm_19':
                            {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                             'similarity_measure': 'euclidean',
                             'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                             'n_templates': 10, 'decision_strategy': 'most_similar_template',
                             'similarity_for_group': 'sum_group', 'k_similar_templates': 1, 'n_jobs': -1},
                        'algorithm_20':
                            {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                             'similarity_measure': 'euclidean',
                             'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                             'n_templates': 15, 'decision_strategy': 'most_similar_template',
                             'similarity_for_group': 'sum_group', 'k_similar_templates': 1, 'n_jobs': -1}
                        }

    def set_med_variant(self, parameters):
        med_parameters = deepcopy(parameters)
        med_parameters['template_construction'] = 'med'
        return med_parameters

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    bench = Benchmark()
    writer = XlsxWriter(RESULTS_SHEET)
    gnb = GaussianNB()
    mnm = MultinomialNB()

    mlp1 = MLPClassifier(max_iter=100, activation='logistic')
    mlp2 = MLPClassifier(max_iter=500, activation='logistic')
    mlp3 = MLPClassifier(max_iter=1000, activation='logistic')

    estimators = [("gnb", gnb), ("mnm", mnm), ("mlp1", mlp1), ("mlp2", mlp2), ("mlp3", mlp3)]

    for learning_set in LearningSet:
        if learning_set != LearningSet.mnist:
            print(learning_set)
            bench.fit_and_save_base_estimators(estimators=estimators, learningSet=learning_set)
    """
    for algorithm, parameters in bench.testing_variants.items():
        print(algorithm + "_avg")
        results = bench.cross_validation_score_10_folds(parameters, LearningSet.iris)
        writer.append_results("LearningSet.iris", algorithm + "_avg", results, np.average(results))
        results = bench.cross_validation_score_10_folds(bench.set_med_variant(parameters), LearningSet.iris)
        print(algorithm + "_med")
        writer.append_results("LearningSet.iris", algorithm + "med", results, np.average(results))
    """



