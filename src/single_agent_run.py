import json
import os
import warnings

import gymnasium as gym
import numpy as np
import tensorflow as tf
from colorama import Fore, Style
from gym.wrappers import RecordVideo

from src.Agents import AgentFactory
from src.Metrics.ResultsPlotter import ResultsPlotter
from src.Settings import settings
from src.Wrappers.CustomEnvironmentWrapper import CustomEnvironmentWrapper

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.core")


#     config = {
#         "observation": {
#             "type": "Kinematics",
#             "vehicles_count": 10,
#             "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
#             "features_range": {
#                 "x": [-100, 100],
#                 "y": [-100, 100],
#                 "vx": [-20, 20],
#                 "vy": [-20, 20]
#             },
#             "absolute": True,
#             "order": "sorted",
#             "observe_intentions": True
#         }
#     }

class SingleAgentRunner:
    def __init__(self, env, test_env, agent):
        self.env = env
        self.test_env = test_env
        self.agent = agent
        self.steps = 0
        self.episode = 0
        self.max_steps = settings.TRAINING_STEPS
        self.rp = ResultsPlotter(agent)

        self.rp.save_config_file(multi_agent=False)
        print("Single Agent: ", settings.AGENT_TYPE)
        print("  - Training Steps: ", self.max_steps)

        log_dir = settings.SAVE_DIR + "/Tensorboard"
        self.summary_writer = tf.summary.create_file_writer(log_dir)

        with self.summary_writer.as_default():
            tf.summary.text("Config", json.dumps(self.rp.get_config_dict(), indent='\n'), step=0)

    def train(self):
        while self.steps < self.max_steps:
            done = False
            state, info = self.env.reset(seed=settings.SEED)

            episode_reward = 0
            agent_speed = np.empty(0)
            starting_episode_steps = self.steps
            while not done:
                if settings.AGENT_TYPE == 'ppo':
                    action, value, probability = self.agent.get_action(state)
                else:
                    action, value, probability = self.agent.get_action(state), None, None

                next_state, reward, done, trunc, info = self.env.step(action)

                if settings.AGENT_TYPE == 'ppo':
                    self.agent.store_experience_in_replay_buffer(state, action, value, reward, done, probability)
                else:
                    self.agent.store_experience_in_replay_buffer(state, action, reward, next_state, done)

                self.agent.learn()
                state = next_state
                episode_reward += reward
                agent_speed = np.append(agent_speed, info["agents_speeds"][0])

                if self.steps % settings.PLOT_STEPS_FREQUENCY == 0:
                    optimal_policy_episode_reward, optimal_policy_agent_speed = self.test()
                    self.rp.steps_history = np.append(self.rp.steps_history, self.steps)
                    self.rp.reward_history = np.append(self.rp.reward_history, optimal_policy_episode_reward)
                    self.rp.speed_history = np.append(self.rp.speed_history, optimal_policy_agent_speed)
                    with self.summary_writer.as_default():
                        tf.summary.scalar('Returns', optimal_policy_episode_reward, step=self.steps)
                        for r_avg in [100, 500]:
                            if len(self.rp.reward_history) >= r_avg:
                                tf.summary.scalar('Returns Rolling Average (' + str(r_avg) + ')',
                                                  np.mean(self.rp.reward_history[-r_avg:]),
                                                  step=self.steps)
                                tf.summary.scalar('Speed Rolling Average (' + str(r_avg) + ')',
                                                  np.mean(self.rp.speed_history[-r_avg:]),
                                                  step=self.steps)
                        self.summary_writer.flush()

                self.steps += 1
            self.episode += 1

            # Output episode rewards and overall status
            print("Episode: ", self.episode)
            print("  - Reward: ", episode_reward)
            print("  - Total Steps: ", self.steps, "/", self.max_steps)
            print("  - Episode Steps: ", self.steps - starting_episode_steps)
            print("  - Max Optimal Policy Reward: ", np.max(self.rp.reward_history))
            if len(self.rp.reward_history) >= 100:
                print("  - Rolling Average (100 optimal policy tests): ", np.mean(self.rp.reward_history[-100:]))
            if len(self.rp.reward_history) >= 500:
                print("  - Rolling Average (500 optimal policy tests): ", np.mean(self.rp.reward_history[-500:]))

    def test(self):
        done = False
        state, info = self.test_env.reset(seed=settings.SEED)

        episode_steps = 0
        episode_reward = 0
        agent_speed = np.empty(0)
        print(Fore.GREEN + "-------------" + Style.RESET_ALL)
        print(Fore.GREEN + "Testing Optimal Policy: ", self.steps / settings.PLOT_STEPS_FREQUENCY, Style.RESET_ALL)
        while not done:
            action = self.agent.get_action(state, training=False)
            state, reward, done, trunc, info = self.test_env.step(action)

            episode_reward += reward
            agent_speed = np.append(agent_speed, info["agents_speeds"][0])
            episode_steps += 1

        avg_speed = np.mean(agent_speed)
        print(Fore.GREEN + " - Steps: ", episode_steps, Style.RESET_ALL)
        print(Fore.GREEN + " - Reward: ", episode_reward, Style.RESET_ALL)
        print(Fore.GREEN + " - Agent Speed: ", avg_speed, Style.RESET_ALL)
        print(Fore.GREEN + "-------------\n" + Style.RESET_ALL)

        return episode_reward, avg_speed

    def save_final_results(self):
        self.rp.save_final_results(self.episode)
        print("Results saved to: ", settings.SAVE_DIR)


def run_single_agent():
    env = initialise_env()
    test_env = initialise_env(record_env=False)

    agent = AgentFactory.create_new_agent(env)
    single_agent_runner = SingleAgentRunner(env, test_env, agent)
    single_agent_runner.train()
    single_agent_runner.save_final_results()
    single_agent_runner.test()


def initialise_env(record_env=True):
    env = gym.make('highway-with-obstructions-v0', render_mode='rgb_array')
    env = CustomEnvironmentWrapper(env)
    env = record_wrap(env) if settings.RECORD_EPISODES[0] and record_env else env
    return env


def record_wrap(env):
    video_folder = settings.SAVE_DIR + "/Videos"
    os.makedirs(video_folder, exist_ok=True)

    env = RecordVideo(
        env,
        video_folder,
        episode_trigger=lambda x: x % settings.RECORD_EPISODES[1] == 0,
        name_prefix="optimal-policy"
    )
    env.unwrapped.set_record_video_wrapper(env)
    return env


if __name__ == "__main__":
    run_single_agent()
