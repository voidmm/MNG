import time
import igraph as ig
import markovgame
import igraph2pykov
import numpy as np
from plot import Plot
from createagentmemory import createagentmemory
from multiprocessing import Pool, cpu_count



CPU_COUNT = cpu_count()

params = eval(open('parameters.txt').read())

number_of_repeats = params['number_of_repeats']
number_of_rounds = params['number_of_rounds']
number_of_agents = params['number_of_agents']
reward = params['reward']

plot_time_intervals = params['plot_time_intervals']
plot_time_points = number_of_rounds / plot_time_intervals

if number_of_repeats % CPU_COUNT != 0:
    number_of_repeats -= number_of_repeats % CPU_COUNT

word_frequencies, initial_word_memory, initial_word_transitions, states = createagentmemory('parameters.txt')

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

    end = time.time()
    print('Required time, in seconds: '+ str(end - start))

    Plot = Plot()
    Plot.plot_sigmoid(history)
    Plot.plot_box_word_probabilities([i[0] for i in initial_word_memory], word_game_counts)
    Plot.plot_word_frequency([i[0] for i in initial_word_memory], word_frequencies)

    #Plot.save_word_frequencies_to_archive(word_frequencies,word_associations,word_game_counts,states)
