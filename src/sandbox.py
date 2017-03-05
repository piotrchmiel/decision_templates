from src.dataset import DataProvider, LearningSet
from src.decision_templates import DecisionTemplatesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.ensemble import BaggingClassifier
lr = LogisticRegression()
knn = KNeighborsClassifier()
gnb = GaussianNB()
dectempl_avg = DecisionTemplatesClassifier(estimators=[('lr', lr), ('knn', knn), ('gnb', gnb)], template_construction='avg',
                                           template_fit_strategy='bootstrap', n_templates=5)
provider = DataProvider()
X, y, _ = provider.get(LearningSet.iris)

dectempl_avg.fit(X, y)