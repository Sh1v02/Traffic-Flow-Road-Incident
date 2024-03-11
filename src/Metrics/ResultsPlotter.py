import os

import numpy as np
from matplotlib import pyplot as plt

from src.Settings import settings


class ResultsPlotter:
    def __init__(self):
        self.steps_history = np.empty(0)
        self.reward_history = np.empty(0)
        self.speed_history = np.empty(0)

    def save_final_results(self, episode_reached):
        self.plot_graph(self.steps_history, self.reward_history, "rewards", save_dir=settings.SAVE_DIR)
        self.plot_graph(self.steps_history, self.speed_history, "speed_history", save_dir=settings.SAVE_DIR,
                        labels=['Steps', 'Speed'])
        r_avg_window = 100
        if episode_reached >= r_avg_window:
            self.plot_graph(self.steps_history[r_avg_window - 1:],
                            np.convolve(self.reward_history, np.ones(r_avg_window) / r_avg_window,
                                        mode='valid'),
                            "r_avg=" + str(r_avg_window), save_dir=settings.SAVE_DIR)
            self.plot_graph(self.steps_history[r_avg_window - 1:],
                            np.convolve(self.speed_history, np.ones(r_avg_window) / r_avg_window, mode='valid'),
                            "speed_history_r_avg=" + str(r_avg_window), save_dir=settings.SAVE_DIR,
                            labels=['Steps', 'Speed'])

        # Save values to text file: rewards.txt
        np.savetxt(settings.SAVE_DIR + "/rewards.txt",
                   (self.steps_history, self.reward_history, self.speed_history), delimiter=',', fmt='%d')

    @staticmethod
    def plot_graph(x_values, y_values, name, save_dir=None, labels=None):
        labels = labels if labels else ["Frames", "Reward"]
        frames_per_step = 15
        plt.figure(figsize=(8, 4))
        # plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
        plt.plot(x_values * 15, y_values, linestyle='-', color='b', linewidth=1.0)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_dir + '/' + name + '.png')
