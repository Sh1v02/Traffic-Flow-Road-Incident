import os
import warnings

import gymnasium as gym
import numpy as np
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
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.steps_history = np.empty(0)
        self.reward_history = np.empty(0)
        self.speed_history = np.empty(0)
        self.steps = 0
        self.episode = 0
        self.max_steps = settings.TRAINING_STEPS
        self.rp = ResultsPlotter(agent)

        self.rp.save_config_file(multi_agent=False)
        print("Single Agent: ", settings.AGENT_TYPE)
        print("  - Training Steps: ", self.max_steps)

    def train(self):
        while self.steps < self.max_steps:
            done = False
            state, info = self.env.reset(seed=settings.SEED)

            episode_reward = 0
            agent_speed = np.empty(0)
            starting_episode_steps = self.steps
            print("seed =", settings.SEED, " - Episode: ", self.episode)
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
                self.steps += 1

            # Add the episode results to the agent's results plotter
            self.episode += 1
            self.rp.steps_history = np.append(self.rp.steps_history, self.steps)
            self.rp.reward_history = np.append(self.rp.reward_history, episode_reward)
            self.rp.speed_history = np.append(self.rp.speed_history, np.mean(agent_speed))

            # Output episode rewards and overall status
            print("  - Reward: ", episode_reward)
            print("  - Total Steps: ", self.steps, "/", self.max_steps)
            print("  - Episode Steps: ", self.steps - starting_episode_steps)
            print("  - Max Reward: ", np.max(self.rp.reward_history))
            if self.episode >= 100:
                print("  - Rolling Average (100 episodes): ", np.mean(self.rp.reward_history[-100:]))
            if self.episode >= 500:
                print("  - Rolling Average (500 episodes): ", np.mean(self.rp.reward_history[-500:]))

    def test(self):
        done = False
        state, info = self.env.reset(seed=settings.SEED)

        episode_steps = 0
        episode_reward = 0
        print("-------------")
        print("Testing optimal policy")
        while not done:
            if settings.AGENT_TYPE == 'ppo':
                action, value, probability = self.agent.get_action(state)
            else:
                action, value, probability = self.agent.get_action(state), None, None

            state, reward, done, trunc, info = self.env.step(action)

            episode_reward += reward
            episode_steps += 1
        print(" - Steps: ", episode_steps)
        print(" - Reward: ", episode_reward)

    def save_final_results(self):
        self.rp.save_final_results(self.episode)
        print("Results saved to: ", settings.SAVE_DIR)


def run_single_agent():
    env = initialise_env()

    agent = AgentFactory.create_new_agent(env)
    single_agent_runner = SingleAgentRunner(env, agent)
    single_agent_runner.train()
    single_agent_runner.save_final_results()
    single_agent_runner.test()


def initialise_env():
    env = gym.make('highway-with-obstructions-v0', render_mode='rgb_array')
    env = CustomEnvironmentWrapper(env)
    env = record_wrap(env) if settings.RECORD_EPISODES[0] else env
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
