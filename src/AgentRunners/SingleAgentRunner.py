import time

import numpy as np
from colorama import Fore, Style

from src.AgentRunners import AgentRunner
from src.Utilities import settings
from src.Utilities.Helper import Helper


class SingleAgentRunner(AgentRunner):
    def __init__(self, env, test_env, agent):
        super().__init__(env, test_env, agent, multi_agent=False)
        self.agent = agent

        Helper.output_information("Single Agent: " + settings.AGENT_TYPE)
        Helper.output_information("  - Training Steps: " + str(self.max_steps))

    def train(self):
        self.start_time = time.time()
        print("\n\nBeginning Training")
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

                if self.steps % settings.PLOT_STEPS_FREQUENCY == 0:
                    optimal_policy_reward, optimal_policy_speed = self.test()
                    self.store_optimal_policy_results(optimal_policy_reward, optimal_policy_speed)

                state = next_state
                episode_reward += reward
                agent_speed = np.append(agent_speed, info["agents_speeds"][0])
                self.steps += 1
            self.episode += 1

            # Output episode results
            self.output_episode_results(episode_reward, self.steps - starting_episode_steps)

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
