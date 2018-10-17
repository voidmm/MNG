import numpy as np
import random
from memoryfunctions import markov_update_words
import pykov
import createagentmemory
import time
from copy import deepcopy



def language_game(number_of_agents, number_of_rounds, plot_time_points, word_frequencies, initial_word_memory,
				 initial_word_transitions, reward):

	history = []
	agents = {}

	for agent_int in range(number_of_agents):
		agents[str(agent_int)] = {'id': agent_int, 
								  'word memory': deepcopy(initial_word_memory),
								  'word transitions': pykov.Chain(deepcopy(initial_word_transitions))}


	def update_association_strength(agent, word, speaker_association, reward):
		successor_states = agent['word transitions'].succ(word)
		transition_probabilities = markov_update_words(successor_states, speaker_association, reward)
		for state, value in transition_probabilities.items():
			agent['word transitions'][word, state] = value


	def communicate(speaker, hearer, word, speaker_association, hearer_association):
		success = 0
		if hearer_association != speaker_association:
			pass
		else:
			success = 1
			update_association_strength(hearer, word, speaker_association, reward)
			update_association_strength(speaker, word, speaker_association, reward)
			speaker['word memory'] = markov_update_words(speaker['word memory'], speaker_association, reward)
			hearer['word memory'] = markov_update_words(hearer['word memory'], speaker_association, reward)
		return success


	def communicate_teach(speaker, hearer, word, speaker_association, hearer_association):
		success = 0
		if hearer_association != speaker_association:
			update_association_strength(hearer, word, speaker_association, reward)
			hearer['word memory'] = markov_update_words(hearer['word memory'], speaker_association, reward)
		else:
			success = 1
			update_association_strength(hearer, word, speaker_association, reward)
			update_association_strength(speaker, word, speaker_association, reward)
			speaker['word memory'] = markov_update_words(speaker['word memory'], speaker_association, reward)
			hearer['word memory'] = markov_update_words(hearer['word memory'], speaker_association, reward)
		return success


	for i in range(number_of_rounds):

		random_agents = random.sample(range(number_of_agents),2)
		speaker_id = random_agents[0]
		hearer_id = random_agents[1]
		speaker = agents[str(speaker_id)]
		hearer = agents[str(hearer_id)]

		word = speaker['word memory'].choose()

		if i % plot_time_points == 0:
			word_index = word_frequencies['word indices'][word]
			index = int(i/plot_time_points)
			word_frequencies['word frequencies'][word_index][index]+=1

		speaker_association = speaker['word transitions'].move(word)
		hearer_association = hearer['word transitions'].move(word)

		success = communicate(speaker, hearer, word, speaker_association, hearer_association)

		history.append(success)

		agent_memories = []

		word_frequencies['word associations'][word_frequencies['word indices'][word]][
			word_frequencies['word indices'][speaker_association]] += 1

	agent_memory_matrix = [np.zeros((len(initial_word_memory),len(initial_word_memory))) for i in range(number_of_agents)]

	for agent_id, agent_values in agents.items():
		word_memory = agent_values['word memory'].values()
		agent_memories.append(list(word_memory))
	word_frequencies['word game counts'].append(np.mean(agent_memories, axis=0))
	
	return history, word_frequencies









