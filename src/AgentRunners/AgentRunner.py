import json
from abc import ABC, abstractmethod
import time

import numpy as np
import tensorflow as tf

from src.Metrics.ResultsPlotter import ResultsPlotter
from src.Utilities import settings
from src.Utilities.Constants import DEVICE
from src.Utilities.Helper import Helper


class AgentRunner(ABC):
    # Assumes that the child class has agents of only one type (eg: all PPO)
    def __init__(self, env, test_env, agent):
        self.env = env
        self.test_env = test_env
        self.steps = 0
        self.episode = 0
        self.max_steps = settings.TRAINING_STEPS

        self.start_time = time.time()

        self.rp = ResultsPlotter(agent)
        self.rp.save_config_file()

        Helper.output_information("Device: " + DEVICE)

        if settings.LOG_TENSORBOARD:
            log_dir = settings.SAVE_DIR + "/Tensorboard"
            self.summary_writer = tf.summary.create_file_writer(log_dir)
            with self.summary_writer.as_default():
                tf.summary.text("Config", json.dumps(self.rp.get_config_dict(), indent='\n'), step=0)

    def output_episode_results(self, episode_reward, episode_steps):
        self.output_remaining_time(100)
        # Output episode rewards and overall status
        print("Episode: ", self.episode)
        print("  - Reward: ", episode_reward)
        print("  - Total Steps: ", self.steps, "/", self.max_steps)
        print("  - Episode Steps: ", episode_steps)
        print("  - Max Optimal Policy Reward: ", np.max(self.rp.reward_history))
        if len(self.rp.reward_history) >= 100:
            print("  - Rolling Average (100 optimal policy tests): ", np.mean(self.rp.reward_history[-100:]))
        if len(self.rp.reward_history) >= 500:
            print("  - Rolling Average (500 optimal policy tests): ", np.mean(self.rp.reward_history[-500:]))

    def store_optimal_policy_results(self, optimal_policy_reward, optimal_policy_speed, plot_names=None):
        plot_names = plot_names if plot_names else ['Returns', 'Speed']
        r_avgs = [100, 500]

        self.rp.steps_history = np.append(self.rp.steps_history, self.steps)
        self.rp.reward_history = np.append(self.rp.reward_history, optimal_policy_reward)
        self.rp.speed_history = np.append(self.rp.speed_history, optimal_policy_speed)

        if settings.LOG_TENSORBOARD:
            with self.summary_writer.as_default():
                for r_avg in r_avgs:
                    if len(self.rp.reward_history) >= r_avg:
                        tf.summary.scalar(plot_names[0] + ' Rolling Average (' + str(r_avg) + ')',
                                          np.mean(self.rp.reward_history[-r_avg:]),
                                          step=self.steps)
                        tf.summary.scalar(plot_names[1] + ' Rolling Average (' + str(r_avg) + ')',
                                          np.mean(self.rp.speed_history[-r_avg:]),
                                          step=self.steps)
                self.summary_writer.flush()

    def save_final_results(self):
        self.rp.save_final_results(self.episode)
        print("Results saved to: ", settings.SAVE_DIR)

    def output_remaining_time(self, steps_to_estimate_from=1000):
        if self.steps >= steps_to_estimate_from:
            time_so_far = time.time() - self.start_time
            multiplier = (self.max_steps - self.steps) / self.steps
            time_remaining = time_so_far * multiplier
            Helper.output_information("Estimated Time Remaining: " + str(time_remaining / 60) + " minutes = " + str(
                time_remaining / 3600) + " hours")

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def test(self):
        raise NotImplementedError
