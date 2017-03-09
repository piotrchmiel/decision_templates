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
    kddcup = 8
    letter = 9
    mnist = 10
    poker = 11
    satimage = 12
    segment = 13
    shuttle = 14
    vehicle = 15
    vowel = 16
    wine = 17
    yeast = 18