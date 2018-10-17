import time
import igraph as ig
import markovgame
import igraph2pykov
import numpy as np
from plot import Plot
from createagentmemory import createagentmemory
from multiprocessing import Pool, cpu_count


"""

Define parameters of the Probabilistic Naming Game

insert the path where the plots of the success history should be saved
decide whether you want to save the figure
the number of agents that partake in each game
the number of rounds defined the number of rounds of one probabilistic naming game
the reward that is applied to the probability of a word that was used by both agents to
name the object
the number of repeats defines the number of total, individual games that are played
the number of times two disjunct communities will be connected
"""

CPU_COUNT = cpu_count()

plot_path = 'reward_progressions/'

graph_path = 'graphexports/'

language_construct_filename = '9wattsstrogatz00'

save_figure = False

number_of_agents = 100

number_of_rounds = 1000

reward = 5

number_of_repeats = 50

if number_of_repeats % CPU_COUNT != 0:
    number_of_repeats -= number_of_repeats % CPU_COUNT

plot_time_intervals = 10
plot_time_points = number_of_rounds / plot_time_intervals

word_frequencies, initial_word_memory, initial_word_transitions, states = createagentmemory(graph_path, language_construct_filename, 
                                                                    number_of_rounds, plot_time_intervals)


def games(number_of_repeats, word_frequencies):
    aggregated_history = []

    for i in range(number_of_repeats):
        history, word_frequencies = markovgame.language_game(number_of_agents, number_of_rounds, plot_time_points,
                                                            word_frequencies, initial_word_memory,
                                                            initial_word_transitions,reward)
        aggregated_history.append(history)
    return aggregated_history, word_frequencies


if __name__ == "__main__":
    start = time.time()

    p = Pool()
    size_of_game_batch = number_of_repeats / CPU_COUNT
    all_parallel_games = [int(size_of_game_batch) for i in range(CPU_COUNT)]
    output = [p.apply_async(games, args=(x, word_frequencies)) for x in all_parallel_games]
    p.close()
    p.join()

    history = [p.get()[0] for p in output]
    history = np.reshape(np.ravel(history, order='A'), (number_of_repeats, number_of_rounds))

    word_frequencies = np.sum([p.get()[1]['word frequencies'] for p in output], axis=0)
    word_associations = [p.get()[1]['word associations'] for p in output]
    word_game_counts = [p.get()[1]['word game counts'] for p in output]
    word_game_counts = np.reshape(word_game_counts, (number_of_repeats, len(initial_word_memory)))

    filename = 'word frequency data/' + 'nonconv' + str(number_of_agents) + ', ' + str(number_of_repeats) + ', ' + str(
        number_of_rounds) + ', ' + str(reward) + ', ' + language_construct_filename
    np.savez(filename + '.npz', name1=word_frequencies, name2=word_associations, name3=word_game_counts, name4=states)

    end = time.time()
    print('Required time, in seconds: '+ str(end - start))

    Plot = Plot(history, save_figure, plot_path, number_of_agents, number_of_repeats,
                  number_of_rounds, plot_time_intervals,
                  reward, language_construct_filename)
    
    Plot.plot_sigmoid()
    Plot.plot_box_word_probabilities([i[0] for i in initial_word_memory], word_game_counts)
    Plot.plot_word_frequency([i[0] for i in initial_word_memory], word_frequencies)
