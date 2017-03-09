from src.dataset import DataProvider
from src.learning_set import LearningSet
from src.decision_templates import DecisionTemplatesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

lr = LogisticRegression()
knn = KNeighborsClassifier()
gnb = GaussianNB()
provider = DataProvider()
X, y, _ = provider.get(LearningSet.iris)

le_ = LabelEncoder()
le_.fit(y)
classes_ = le_.classes_
transformed_y = le_.transform(y)

lr.fit(X, transformed_y)
knn.fit(X, transformed_y)
gnb.fit(X, transformed_y)

dectempl_avg = DecisionTemplatesClassifier(estimators=[('lr', lr), ('knn', knn), ('gnb', gnb)], template_construction='avg',
                                           template_fit_strategy='bootstrap', n_templates=5)

dectempl_avg.fit(X, y)