import pykov
import numpy as np
from multiprocessing import Pool
from collections import OrderedDict


def update_word_probabilities(word_probabilities, index, reward):
    word_probabilities[index] += reward
    summ = sum(word_probabilities)
    corrected_word_probabilities = list(map(lambda i: float(i) / summ, word_probabilities))

    return corrected_word_probabilities


def normalize(vector):
    summ = sum(vector)
    return list(map(lambda i: float(i) / summ, vector))


def markov_update_words(pykov_words, association, reward):
    pykov_words[association] = pykov_words[association] + reward
    summ = sum(pykov_words.values())
    normalized_probabilities = list(map(lambda i: float(i) / summ, pykov_words.values()))

    for i in range(len(normalized_probabilities)):
        if normalized_probabilities[i] < 4.768371584931426e-04:
            normalized_probabilities[i] = 0.001


    normalized_probabilities = OrderedDict(zip(pykov_words.keys(), normalized_probabilities))

    return pykov.Vector(normalized_probabilities)
