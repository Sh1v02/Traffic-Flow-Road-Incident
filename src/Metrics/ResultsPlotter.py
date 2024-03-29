import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.Utilities import settings, graphics_settings, multi_agent_settings
from src.Utilities.Helper import Helper


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
            if len(self.reward_history) >= r_avg_window:
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

    def get_config_dict(self):
        config = {
            "TEST_TYPE": settings.RUN_TYPE,
            "AGENT_TYPE": str(settings.AGENT_TYPE.upper()),
        }

        if settings.RUN_TYPE.lower() == "multiagent":
            config.update(
                {
                    "AGENT_COUNT": multi_agent_settings.AGENT_COUNT,
                    "OBSTRUCTION_COUNT": graphics_settings.OBSTRUCTION_COUNT,
                    "WAIT_UNTIL_TERMINATED": str(multi_agent_settings.WAIT_UNTIL_ALL_AGENTS_TERMINATED),
                    "TEAM_SPIRIT": str(multi_agent_settings.TEAM_SPIRIT),
                    "SHARED_REPLAY_BUFFER": str(multi_agent_settings.SHARED_REPLAY_BUFFER),
                    "PARAMETER_SHARING": str(multi_agent_settings.PARAMETER_SHARING)
                }
            )

        config.update(
            {
                "ENVIRONMENT_SEED": str(settings.ENVIRONMENT_SEED),
                "SEED": str(settings.SEED),
                "TRAINING_STEPS": str(settings.TRAINING_STEPS),
                "PLOT_STEPS_FREQUENCY": str(settings.PLOT_STEPS_FREQUENCY),
                "LANE_COUNT": str(graphics_settings.LANE_COUNT),
                "OBSTRUCTION_COUNT": str(graphics_settings.OBSTRUCTION_COUNT)
            }
        )

        config.update(self.agent.get_agent_specific_config())

        return config

    def save_config_file(self):
        os.makedirs(settings.SAVE_DIR, exist_ok=True)
        save_path = settings.SAVE_DIR + "/config.txt"
        config = self.get_config_dict()
        config_df = pd.DataFrame(list(config.items()), columns=['Setting', 'Value'])
        config_df.to_csv(save_path, index=False)
        Helper.output_information("Config Saved to: " + save_path)

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
