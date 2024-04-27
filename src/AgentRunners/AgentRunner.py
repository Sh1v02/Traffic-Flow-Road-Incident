import json
import time
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import torch

from src.Metrics.ResultsPlotter import ResultsPlotter
from src.Utilities import settings, multi_agent_settings
from src.Utilities.Constants import DEVICE
from src.Utilities.Helper import Helper
from src.Wrappers.GPUSupport import tensor


class AgentRunner(ABC):
    # Assumes that the child class has agents of only one type (eg: all PPO)
    def __init__(self, env, test_env, agent, global_state_dims=0):
        self.env = env
        self.test_env = test_env
        self.steps = 0
        self.episode = 0
        self.max_steps = settings.TRAINING_STEPS

        self.team_spirit_tau = multi_agent_settings.TEAM_SPIRIT[1]
        self.interpolate_team_spirit = self.team_spirit_tau < multi_agent_settings.TEAM_SPIRIT[2]
        steps_to_max_team_spirit = multi_agent_settings.TEAM_SPIRIT[3]
        self.interpolate_team_spirit_rate = (multi_agent_settings.TEAM_SPIRIT[
                                                 2] - self.team_spirit_tau) / steps_to_max_team_spirit

        self.agent_type = settings.AGENT_TYPE.lower()
        self.global_state_dims = global_state_dims
        self.vf_input_representation = (
            settings.QMIX_VALUE_FUNCTION_INPUT_REPRESENTATION if self.agent_type == "qmix" else (
                settings.MAPPO_VALUE_FUNCTION_INPUT_REPRESENTATION)).lower()

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
        print("Episode: ", self.episode, " - Total Steps: ", self.steps, "/", self.max_steps)
        print("  - Reward: ", episode_reward)
        print("  - Episode Steps: ", episode_steps)
        print("  - Max Optimal Policy Reward: ", np.max(self.rp.reward_history))
        print("  - Trunc Count: ", self.rp.trunc_count)
        if len(self.rp.reward_history) >= 100:
            print("  - Rolling Average (100 optimal policy tests): ", np.mean(self.rp.reward_history[-100:]))
        if len(self.rp.reward_history) >= 500:
            print("  - Rolling Average (500 optimal policy tests): ", np.mean(self.rp.reward_history[-500:]))

    def store_optimal_policy_results(self, optimal_policy_reward, optimal_policy_speed, trunc, plot_names=None):
        plot_names = plot_names if plot_names else ['Returns', 'Speed']
        r_avgs = [100, 500]

        self.rp.steps_history = np.append(self.rp.steps_history, self.steps)
        self.rp.reward_history = np.append(self.rp.reward_history, optimal_policy_reward)
        self.rp.speed_history = np.append(self.rp.speed_history, optimal_policy_speed)
        self.rp.trunc_count += int(trunc)
        self.rp.trunc_history = np.append(self.rp.trunc_history, self.rp.trunc_count)

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
        self.rp.save_final_results()
        print("Results saved to: ", settings.SAVE_DIR)

    def output_remaining_time(self, steps_to_estimate_from=1000):
        if self.steps >= steps_to_estimate_from:
            time_so_far = time.time() - self.start_time
            multiplier = (self.max_steps - self.steps) / self.steps
            time_remaining = time_so_far * multiplier
            Helper.output_information("Estimated Time Remaining: " + str(time_remaining / 60) + " minutes = " + str(
                time_remaining / 3600) + " hours")

    def update_global_states(self, local_states, global_states, dones):
        global_state = self.env.get_global_state(local_states)
        # Update the global_states
        for agent_index in range(len(dones)):
            # If value function death masking (set the global state to 0 here)
            if multi_agent_settings.VALUE_FUNCTION_DEATH_MASKING and dones[agent_index]:
                if self.agent_type == "qmix":
                    global_states[agent_index] = tensor(np.zeros(self.global_state_dims))
                else:
                    global_states[agent_index] = np.zeros(self.global_state_dims)
            else:
                if self.vf_input_representation == "as":
                    if self.agent_type == "qmix":
                        global_states[agent_index] = torch.cat((global_state, local_states[agent_index]), dim=0)
                    else:
                        global_states[agent_index] = np.concatenate((global_state, local_states[agent_index]),
                                                                    axis=0)
                else:
                    global_states[agent_index] = global_state
        return global_states

    def calculate_team_spirit_rewards(self, individual_rewards, team_reward):
        if self.team_spirit_tau >= 1.0:
            return individual_rewards

        team_spirited_rewards = tuple(
            ((1 - self.team_spirit_tau) * reward) + (self.team_spirit_tau * team_reward)
            for reward in individual_rewards
        )
        self.team_spirit_tau = self.team_spirit_tau if not self.interpolate_team_spirit else (
                self.team_spirit_tau + self.interpolate_team_spirit_rate)
        return team_spirited_rewards

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def test(self):
        raise NotImplementedError
