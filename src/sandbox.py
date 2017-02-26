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
dectempl_avg = DecisionTemplatesClassifier(estimators=[('lr', lr), ('knn', knn), ('gnb', gnb)], template_creation='avg')
provider = DataProvider()
X, y, _ = provider.get(LearningSet.iris)

BaggingClassifier().fit(X, y)
for i in dectempl_avg._random_set(5, False, 40, np.ones((40,))):
    print(i)