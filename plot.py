import pylab
import string
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from statsmodels.nonparametric.smoothers_lowess import lowess


class Plot():

    def __init__(self, history, save_figure, path, number_of_agents, number_of_repeats,
                 number_of_rounds, plot_time_intervals, reward, language_construct_filename):
        self.history = history
        self.save_figure = save_figure
        self.path = path
        self.number_of_agents = number_of_agents
        self.number_of_repeats = number_of_repeats
        self.number_of_rounds = number_of_rounds
        self.plot_time_intervals = plot_time_intervals
        self.reward = reward
        self.extrapolation_factor = int(np.ceil(self.number_of_rounds / 10000))
        self.language_construct_filename = language_construct_filename


    def sigmoid(self, x, a, b, c):
        np.seterr(all='warn')
        return 1 / (a * np.exp(-b * x) + 1)


    def fit_sigmoid(self, x, y):
        popt, pcov = curve_fit(self.sigmoid, x, y)
        a, b, c = popt
        return a, b, c


    def nonconvergent_sigmoid(self, x, a, b, c):
        np.seterr(all='warn')
        return c / (a * np.exp(-b * x) + 1)


    def fit_nonconvergent_sigmoid(self, x, y):
        popt, pcov = curve_fit(self.nonconvergent_sigmoid, x, y)
        np.seterr(all='warn')
        return popt


    def plot_sigmoid(self):
        original_number_of_rounds = self.number_of_rounds
        mean_history = np.sum(self.history, axis=0)

        if self.extrapolation_factor >= 2:
            mean_history = mean_history[0::self.extrapolation_factor]
            self.number_of_rounds = len(mean_history)

        total_number_of_games = np.multiply(np.ones(self.number_of_rounds), self.number_of_repeats)

        y = np.divide(mean_history, total_number_of_games)
        x = np.linspace(0, original_number_of_rounds, self.number_of_rounds)

        fig, ax = plt.subplots(figsize=(12, 6))
        ml = AutoMinorLocator()
        ax.plot(x, y, 'silver')
        #filtered = lowess(y, x, is_sorted=True, frac=0.1, it=0)
        popt, pcov = self.fit_nonconvergent_sigmoid(x,y)
        ax.plot(x, self.nonconvergent_sigmoid(x, *popt), color='navy', linewidth=1)
        print(popt)
        #ax.plot(filtered[:, 0], filtered[:, 1], 'navy', linewidth=0.8)

       
        ax.set_xlabel('$t$')
        ax.set_ylabel('$S(t)$', rotation=90)
        ax.xaxis.set_minor_locator(ml)
        ax.legend(('averaged success', 'success'))
        ax.set_yticks(np.arange(0, 1.25, 0.25))

        kwargs = {'type': 'sigmoid'}

        if self.save_figure:
            self.save_markov_figure(**kwargs)
            self.save_markov_to_archive(np.linspace(0, original_number_of_rounds, self.number_of_rounds), y, original_number_of_rounds)
        plt.show()


    def plot_association_frequency(self, states, word_associations):
        sum_associations = word_associations[0]
        for single_game_associations in word_associations[1:]:
            sum_associations = np.add(sum_associations, single_game_associations)

        fig, ax = plt.subplots(figsize=(5, 5))
        im = plt.imshow(sum_associations, aspect='auto', interpolation="nearest", origin="upper", cmap='PuBu')
        #ax.title('association frequencies of the last round')
        plt.colorbar(im, spacing='uniform', drawedges='True')
        plt.xticks(np.linspace(0, len(states) - 1, len(states)), list(states), rotation='horizontal')
        plt.yticks(np.linspace(0, len(states) - 1, len(states)), list(states), rotation='horizontal')

        kwargs = {'words': len(states)}
        if self.save_figure:
            self.save_markov_figure(**kwargs)
        plt.show()


    def plot_word_frequency(self, states, word_frequencies):
        round_number = np.shape(word_frequencies)[1]
        round_ticks = np.linspace(10, self.number_of_rounds, round_number, dtype=int)

        plt.figure()
        im = plt.imshow(word_frequencies, aspect='auto', interpolation='nearest', origin='upper', cmap='Blues')
        plt.title('word frequencies for each round')
        plt.colorbar(im, spacing='uniform', drawedges='True')
        plt.xticks(np.linspace(0, len(round_ticks) - 1, len(round_ticks)), round_ticks)
        plt.yticks(np.linspace(0, len(states) - 1, len(states)), list(states), rotation='horizontal')
        
        kwargs = {'words': 'word frequencies'}
        if self.save_figure:
            self.save_markov_figure(**kwargs)
        plt.show()


    def plot_box_word_probabilities(self, states, word_game_counts):
        _, ax = plt.subplots(figsize=(5,4))
        df = pd.DataFrame(word_game_counts, columns=states)
        df.boxplot(column=states, ax=ax)
        ax.set_ylabel('word probabilities')
        ax.set_xlabel('words')
        #ax.set_title('mean word probabilities over all games')

        kwargs = {'game counts': 'game counts'}
        if self.save_figure:
            self.save_markov_figure(**kwargs)
        plt.show()


    def plot_box_word_probabilities_of_connects(self, states, pre_connects0, pre_connects1,
                                                post_connects0, post_connects1):
        df00 = pd.DataFrame(pre_connects0[0][0], columns=states)
        df01 = pd.DataFrame(pre_connects1[0][0], columns=states)
        df10 = pd.DataFrame(post_connects0[0][0], columns=states)
        df11 = pd.DataFrame(post_connects1[0][0], columns=states)

        ax = {'0': [0, 0], '1': [1, 0],
              '2': [0, 1], '3': [1, 1]}

        ax[0, 0].set_ylabel('word probabilities, topology 0', fontsize=8)
        ax[1, 0].set_ylabel('word probabilities, topology 1', fontsize=8)
        ax[1, 0].set_xlabel('words', fontsize=8)
        ax[1, 1].set_xlabel('words', fontsize=8)
        ax[0, 1].set_title('after connecting topologies', fontsize=9)
        ax[0, 0].set_title('before connecting topologies', fontsize=9)

        for i, c in enumerate([df00, df01, df10, df11]):
            ax0 = ax[str(i)][0]
            ax1 = ax[str(i)][1]
            c.boxplot(ax=ax[ax0, ax1], showfliers=False)

        kwargs = {'type': 'boxplot'}
        if self.save_figure:
            self.save_markov_figure(**kwargs)
        plt.show()


    def save_markov_figure(self, **kwargs):
        file_string = 'Agents {}, repeats {}, rounds {}, reward {}, graph {}'

        for k, v in kwargs.items():
            file_string += ', ' + ' {' + k + '}'
        if self.save_figure:
            plt.savefig(self.path + file_string.format(self.number_of_agents, self.number_of_repeats,
                                                        self.number_of_rounds, self.reward,
                                                        self.language_construct_filename, **kwargs) + '.png',
                                                        dpi=300)


    def save_markov_to_archive(self, x, y, plot_time_intervals):
        
        file_name = str(self.path) +'markov' + str(self.number_of_agents) + ', ' + str(self.number_of_repeats) + ', ' \
                             + str(self.number_of_rounds) + ', ' + str(self.reward) + ', ' + str(self.language_construct_filename)
        np.savez(file_name + '.npz', name1=x, name2=y,  name3=plot_time_intervals)
