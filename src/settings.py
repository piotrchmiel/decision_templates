from os import path

BASE_DIR = path.dirname(path.dirname(__file__))

DATA_DIR = path.join(BASE_DIR, "data")

RESULTS_SHEET = path.join(BASE_DIR, "results.xlsx")

PICKLED_ESTIMATORS_DIR = path.join(BASE_DIR, "estimators")