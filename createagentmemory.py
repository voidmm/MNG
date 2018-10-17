from igraph2pykov import GraphtoMarkovMatrix
import numpy as np
import igraph as ig


def createagentmemory(graph_path, language_construct_filename, number_of_rounds, plot_time_intervals):
    language_construct = ig.Graph.Read_Pickle(fname=graph_path + language_construct_filename)
    pyk = GraphtoMarkovMatrix(language_construct, vertex_weights='degree')

    initial_word_memory = pyk.get_pykov_vector()
    initial_word_transitions = pyk.get_symmetric_pykov_matrix()
    states = pyk.get_state_names()

    word_indices = {}
    word_index = 0

    for state in states:
        word_indices[state] = word_index
        word_index += 1

    word_frequencies = {'word indices': word_indices, 
                        'word frequencies': np.zeros((len(states), int(plot_time_intervals))),
                        'word associations': np.zeros((len(states), len(states))),
                        'pre-connect word frequencies topology 0': [],
                        'pre-connect word frequencies topology 1': [],
                        'post-connect word frequencies topology 0': [],
                        'post-connect word frequencies topology 1': [],
                        'word game counts': []}


    return word_frequencies, initial_word_memory, initial_word_transitions, states
