import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.Settings import settings


class ResultsPlotter:
    def __init__(self, agent):
        self.agent = agent
        self.steps_history = np.empty(0)
        self.reward_history = np.empty(0)
        self.speed_history = np.empty(0)

    def save_final_results(self, episode_reached):
        self.plot_graph(self.steps_history, self.reward_history, "rewards", save_dir=settings.SAVE_DIR)
        self.plot_graph(self.steps_history, self.speed_history, "speed_history", save_dir=settings.SAVE_DIR,
                        labels=['Steps', 'Speed'])
        r_avg_windows = [100, 500]
        for r_avg_window in r_avg_windows:
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

    def get_config_dict(self, multi_agent=False):
        config = {
            "TEST_TYPE": "SingleAgent" if not multi_agent else "MultiAgent",
            "AGENT_TYPE": str(settings.AGENT_TYPE.upper()),
            "SEED": str(settings.SEED),
            "TRAINING_STEPS": str(settings.TRAINING_STEPS),
            "DISCOUNT_FACTOR/GAMMA": settings.DISCOUNT_FACTOR
        }
        config.update(self.agent.get_agent_specific_config())

        return config

    def save_config_file(self, multi_agent=False):
        save_path = settings.SAVE_DIR + "/config.txt"
        config = self.get_config_dict(multi_agent)
        config_df = pd.DataFrame(list(config.items()), columns=['Setting', 'Value'])
        config_df.to_csv(save_path, index=False)
        print("Config Saved to: ", save_path)
