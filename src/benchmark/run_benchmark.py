import warnings
import numpy as np

from src.dataset import LearningSet
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
            results = bench.cross_validation_score(parameters=parameters, learning_set=learning_set)
            print(str(learning_set), results)
            writer.append_results(str(learning_set), algorithm + "_avg", results, np.average(results))
            results = bench.cross_validation_score(parameters=set_med_variant(parameters), learning_set=learning_set)
            print(str(learning_set), results)
            writer.append_results(str(learning_set), algorithm + "med", results, np.average(results))
