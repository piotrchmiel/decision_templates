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
    mnist = 8
    satimage = 9
    segment = 10
    shuttle = 11
    vehicle = 12
    vowel = 13
    wine = 14
    yeast = 15
    kddcup = 16
    letter = 17
    poker = 18
