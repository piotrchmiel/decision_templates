from src.dataset import DataProvider, LearningSet
from src.decision_templates import DecisionTemplatesClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection._validation import _score
from sklearn.model_selection import cross_val_score
from typing import List
import numpy as np


class Benchmark(object):
    def __init__(self):
        self.provider = DataProvider()

    def cross_validation_score(self, estimator: DecisionTemplatesClassifier, learningSet: LearningSet,
                               groups: np.ndarray = None, scoring: str = None, cv: int = 10, n_jobs: int = 1,
                               verbose: int = 0, fit_params: dict = None, pre_dispatch: str ='2*n_jobs') -> List[float]:
        try:
            keel_iterator = self.provider.make_k_fold_generator(learningSet, cv)
        except:
            X, y, _ = self.provider.get(learningSet)
            return cross_val_score(estimator, X, y, groups, scoring, cv, n_jobs, verbose, fit_params, pre_dispatch)
        else:
            scorer = check_scoring(estimator, scoring=None)
            parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
            scores = parallel(delayed(self._fit_and_score)(clone(estimator), scorer, train_X, train_y, test_X, test_y)
                          for train_X, train_y, test_X, test_y in keel_iterator)
            return scores

    def _fit_and_score(self, esitmator: DecisionTemplatesClassifier, scorer, train_X, train_y, test_X, test_y):
        esitmator.fit(train_X, train_y)
        return _score(esitmator, test_X, test_y, scorer)


if __name__ == '__main__':
    bench = Benchmark()
    lr = LogisticRegression()
    knn = KNeighborsClassifier()
    gnb = GaussianNB()
    dectempl_avg = DecisionTemplatesClassifier(estimators=[('lr', lr), ('knn', knn), ('gnb', gnb)], template_creation='avg')

    print(bench.cross_validation_score(dectempl_avg, learningSet=LearningSet.vehicle))

    dectempl_med = DecisionTemplatesClassifier(estimators=[('lr', lr), ('knn', knn), ('gnb', gnb)], template_creation='med')

    print(bench.cross_validation_score(dectempl_med, learningSet=LearningSet.vehicle))
