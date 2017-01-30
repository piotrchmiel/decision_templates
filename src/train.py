from src.decision_templates import DecisionTemplatesClassifier
from src.dataset import DataProvider, LearningSet
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

def train():
    provider = DataProvider()
    target, labels = provider.get(LearningSet.iris)
    lr = LogisticRegression()
    knn = KNeighborsClassifier()
    gnb = GaussianNB()
    dectempl = DecisionTemplatesClassifier(estimators=[('lr', lr), ('knn', knn), ('gnb', gnb)])
    dectempl.fit(target, labels)
    print(dectempl.score(target, labels))
    print(dectempl.get_params())
if __name__ == '__main__':
    train()