from enum import unique, Enum


@unique
class LearningSet(Enum):
    abalone = 1
    breast = 2
    cleveland = 3
    dermatology = 4
    ecoli = 5
    flare = 6
    iris = 7
    satimage = 8
    segment = 9
    shuttle = 10
    vehicle = 11
    vowel = 12
    wine = 13
    yeast = 14
    letter = 15
    mnist = 16
    #kddcup = 17
    #poker = 18
