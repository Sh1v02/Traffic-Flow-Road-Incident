import time

import numpy as np
from colorama import Fore, Style

from src.AgentRunners import AgentRunner
from src.Utilities import settings, multi_agent_settings
from src.Utilities.Helper import Helper


class MultiAgentRunner(AgentRunner):
    def __init__(self, env, test_env, agents):
        # Only pass one agent instance in, just to give it a reference agent type to call methods from
        super().__init__(env, test_env, agents[0])
        self.agents = agents
        self.agent_count = len(agents)
        self.until_all_done = True

        Helper.output_information("Multi Agent: " + self.agent_type)
        Helper.output_information("  - Training Steps: " + str(self.max_steps))

    def train(self):
        self.start_time = time.time()

        # Evaluate for the first time:
        optimal_policy_reward, optimal_policy_speed = self.test()
        self.store_optimal_policy_results(optimal_policy_reward, optimal_policy_speed,
                                          ['Team Returns', 'Average Speed'])

        print("\n\nBeginning Training")
        while self.steps < self.max_steps:
            done = False
            states, infos = self.env.reset(seed=settings.SEED)

            episode_reward = 0
            starting_episode_steps = self.steps

            while not done:
                actions, values, probabilities = (), [], []
                if self.agent_type == "ppo":
                    for i in range(self.agent_count):
                        action, value, prob = self.agents[i].get_action(states[i])
                        actions += (action,)
                        values.append(value)
                        probabilities.append(prob)
                else:
                    actions = tuple(self.agents[i].get_action(states[i]) for i in range(self.agent_count))

                next_states, team_reward, done, trunc, infos = self.env.step(actions)

                rewards, dones = infos["agents_rewards"], infos["agents_dones"]

                # If for training, we want to wait for all agents to be done before ending the episode, then update done
                if multi_agent_settings.WAIT_UNTIL_ALL_AGENTS_TERMINATED[0]:
                    done = all(dones)

                # If using team spirit, update the rewards to use the team spirit calculation
                if multi_agent_settings.TEAM_SPIRIT[0]:
                    rewards = self.calculate_team_spirit_rewards(rewards, team_reward)

                # Add to the shared replay buffer (if shared)
                if multi_agent_settings.SHARED_REPLAY_BUFFER:
                    if self.agent_type == "ppo":
                        self.agents[0].store_experience_in_replay_buffer(
                            states, actions, values, rewards, dones, probabilities
                        )
                    else:
                        self.agents[0].store_experience_in_replay_buffer(
                            states, actions, rewards, next_states, dones
                        )
                else:
                    # Add to each agent's replay buffer (if not shared)
                    for i in range(self.agent_count):
                        # TODO: Should the agent be adding to the experience buffer if they remain in a terminal state?
                        if (infos["agents_previous_dones"][i]
                                and multi_agent_settings.DEATH_HANDLING.lower() == "stop_adding"):
                            continue

                        if self.agent_type == "ppo":
                            self.agents[i].store_experience_in_replay_buffer(
                                states[i], actions[i], values[i], rewards[i], dones[i], probabilities[i]
                            )
                        else:
                            self.agents[i].store_experience_in_replay_buffer(
                                states[i], actions[i], rewards[i], next_states[i], dones[i]
                            )
                for agent in self.agents:
                    agent.learn()

                episode_reward += team_reward
                states = next_states
                self.steps += 1

                if self.steps % settings.PLOT_STEPS_FREQUENCY == 0:
                    optimal_policy_reward, optimal_policy_speed = self.test()
                    self.store_optimal_policy_results(optimal_policy_reward, optimal_policy_speed,
                                                      ['Team Returns', 'Average Speed'])
            self.episode += 1

            # Output episode results
            self.output_episode_results(episode_reward, self.steps - starting_episode_steps)

    def test(self):
        done = False
        states, infos = self.test_env.reset(seed=settings.SEED)

        episode_steps = 0
        episode_reward = 0
        avg_agents_speed = np.empty(0)
        print(Fore.GREEN + "-------------" + Style.RESET_ALL)
        print(Fore.GREEN + "Testing Optimal Policy: ", self.steps / settings.PLOT_STEPS_FREQUENCY, Style.RESET_ALL)
        while not done:
            actions = tuple(self.agents[i].get_action(states[i], training=False) for i in range(self.agent_count))
            states, team_rewards, done, trunc, infos = self.test_env.step(actions)

            # If for testing, we want to wait for all agents to be done before ending the episode, then update done
            if multi_agent_settings.WAIT_UNTIL_ALL_AGENTS_TERMINATED[1]:
                done = all(infos["dones"])

            episode_reward += team_rewards
            avg_agents_speed = np.append(avg_agents_speed, sum(infos["agents_speeds"]) / len(infos["agents_speeds"]))
            episode_steps += 1

        avg_speed = np.mean(avg_agents_speed)
        print(Fore.GREEN + " - Steps: ", episode_steps, Style.RESET_ALL)
        print(Fore.GREEN + " - Reward: ", episode_reward, Style.RESET_ALL)
        print(Fore.GREEN + " - Agent Speed: ", avg_speed, Style.RESET_ALL)
        print(Fore.GREEN + "-------------\n" + Style.RESET_ALL)

        return episode_reward, avg_speed

    @staticmethod
    def calculate_team_spirit_rewards(individual_rewards, team_reward):
        tau = multi_agent_settings.TEAM_SPIRIT[1]
        team_spirited_rewards = tuple(((1 - tau) * reward) + (tau * team_reward) for reward in individual_rewards)
        return team_spirited_rewards
