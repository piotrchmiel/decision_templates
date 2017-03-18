import warnings
import numpy as np
from copy import deepcopy
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier

from src.learning_set import LearningSet
from src.writers.excel import XlsxWriter
from src.benchmark.benchmark import Benchmark
from src.settings import RESULTS_SHEET
from src.benchmark.testing_parameters import TESTING_PARAMETERS, make_algorithm_name


def create_esimators():
    gnb = GaussianNB()
    bnb = BernoulliNB()

    mlp1 = MLPClassifier(max_iter=100, activation='logistic')
    mlp2 = MLPClassifier(max_iter=500, activation='logistic')
    mlp3 = MLPClassifier(max_iter=1000, activation='logistic')
    return [("gnb", gnb), ("bnb", bnb), ("mlp1", mlp1), ("mlp2", mlp2), ("mlp3", mlp3)]


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    bench = Benchmark()
    writer = XlsxWriter(RESULTS_SHEET)

    for learning_set in LearningSet:

        for algorithm in ('algorithm_no_group', 'algorithm_groups'):
            for similarity_measure in ('euclidean', 'fuzzy_intersection_divided_by_fuzzy_union',
                                      'symetric_difference_hamming', 'symetric_difference', 'camberra'):
                for template_construction in ['avg', 'med']:
                    learn_parameters = deepcopy(TESTING_PARAMETERS[algorithm])
                    learn_parameters['estimators'] = create_esimators()
                    learn_parameters['similarity_measure'] = similarity_measure
                    learn_parameters['template_construction'] = template_construction

                    results = bench.cross_validation_score(parameters=learn_parameters, learning_set=learning_set,
                                                           n_jobs=-1)
                    print(str(learning_set), make_algorithm_name(algorithm, learn_parameters), results)
                    writer.append_results(str(learning_set), make_algorithm_name(algorithm, learn_parameters),
                                          results, np.average(results))

        for algorithm in ('algorithm_groups_multiple_temp',):
            for similarity_measure in ('euclidean', 'fuzzy_intersection_divided_by_fuzzy_union',
                                      'symetric_difference_hamming', 'symetric_difference', 'camberra'):
                for template_construction in ('avg', 'med'):
                    for template_fit_strategy in ('cluster', 'bootstrap', 'random_subspace'):
                        for n_templates in (3, 5, 10):
                            for similarity_for_group in ('separately', 'average_group', 'sum_group'):
                                learn_parameters = deepcopy(TESTING_PARAMETERS[algorithm])
                                learn_parameters['estimators'] = create_esimators()
                                learn_parameters['similarity_measure'] = similarity_measure
                                learn_parameters['template_construction'] = template_construction
                                learn_parameters['template_fit_strategy'] = template_fit_strategy
                                learn_parameters['n_templates'] = n_templates
                                learn_parameters['similarity_for_group'] = similarity_for_group

                                results = bench.cross_validation_score(parameters=learn_parameters,
                                                                       learning_set=learning_set,
                                                                       n_jobs=-1)
                                print(str(learning_set), make_algorithm_name(algorithm, learn_parameters),
                                      results)
                                writer.append_results(str(learning_set),
                                                      make_algorithm_name(algorithm, learn_parameters),
                                                      results, np.average(results))


