import warnings
import numpy as np
from copy import deepcopy
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier

from src.learning_set import LearningSet
from src.writers.excel import XlsxWriter
from src.benchmark.benchmark import Benchmark
from src.settings import RESULTS_SHEET
from src.benchmark.testing_parameters import TESTING_PARAMETERS, set_med_variant

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    bench = Benchmark()
    writer = XlsxWriter(RESULTS_SHEET)

    for learning_set in LearningSet:
        for algorithm, parameters in TESTING_PARAMETERS.items():
            gnb = GaussianNB()
            bnb = BernoulliNB()

            mlp1 = MLPClassifier(max_iter=100, activation='logistic')
            mlp2 = MLPClassifier(max_iter=500, activation='logistic')
            mlp3 = MLPClassifier(max_iter=1000, activation='logistic')

            estimators = [("gnb", gnb), ("bnb", bnb), ("mlp1", mlp1), ("mlp2", mlp2), ("mlp3", mlp3)]
            learn_parameters = deepcopy(parameters)
            learn_parameters['estimators'] = estimators

            results = bench.cross_validation_score(parameters=learn_parameters, learning_set=learning_set, n_jobs=-1)
            print(str(learning_set), results)
            writer.append_results(str(learning_set), algorithm + "_avg", results, np.average(results))

            results = bench.cross_validation_score(parameters=set_med_variant(learn_parameters),
                                                   learning_set=learning_set, n_jobs=-1)
            print(str(learning_set), results)
            writer.append_results(str(learning_set), algorithm + "med", results, np.average(results))
