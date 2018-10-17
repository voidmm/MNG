import igraph as ig
import pykov
import string
import collections
import itertools
import random


class GraphtoMarkovMatrix():

    def __init__(self, language_construct, vertex_weights):
        self.language_construct = language_construct
        self.number_of_states = language_construct.vcount()
        self.vertex_names = list(string.ascii_lowercase)
        self.states = collections.OrderedDict()
        exec('self.set_states_' + vertex_weights + '()')

    def set_states_uniform(self):
        if self.number_of_states < len(self.vertex_names):
            for string_literal in vertex_names[:self.number_of_states]:
                self.states[string_literal] = 1 / self.number_of_states
        else:
            print('please choose a language construct that has fewer than 24 vertexs')

    def get_state_names(self):
        return list(self.states.keys())

    def get_pykov_vector(self):
    	factor = 1.0 / sum(self.states.values())
    	for k in self.states:
    		self.states[k] = self.states[k] * factor
    	p = pykov.Vector(self.states)
    	return p
    
    def set_states_degree(self):
        vertex_degrees = self.language_construct.vs.degree()

        if self.number_of_states < len(self.vertex_names):
            for string_literal in self.vertex_names[:self.number_of_states]:
                vertex_probability = 1 - (1 / (vertex_degrees[self.vertex_names.index(string_literal)]))
                if vertex_probability == 0:
                    vertex_probability = 0.05
                self.states[string_literal] = vertex_probability
        else:
            print('please choose a language construct that has fewer than 24 vertexs')

    def get_symmetric_pykov_matrix(self):
        states = self.get_state_names()
        edge_dict = collections.OrderedDict()
        edges = self.language_construct.get_edgelist()
        vertex_degrees = self.language_construct.vs.degree()
        transition_probabilities = [1 / degree for degree in vertex_degrees if degree > 0]

        for edge in edges:
            state1 = states[edge[0]]
            state2 = states[edge[1]]
            edge_dict[(state1, state2)] = transition_probabilities[edge[0]]
            edge_dict[(state2, state1)] = transition_probabilities[edge[1]]

        return pykov.Matrix(edge_dict)
