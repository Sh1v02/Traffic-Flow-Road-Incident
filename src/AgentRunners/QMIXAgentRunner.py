import time

import numpy as np
import torch

from src.AgentRunners import AgentRunner
from src.Agents.QMIXAgent import QMIXAgent
from src.Utilities import settings, multi_agent_settings
from src.Utilities.Helper import Helper


class QMIXAgentRunner(AgentRunner):
    def __init__(self, env, test_env, local_state_dims, global_state_dims, action_dims, hidden_layer_dims=64):
        self.agent = QMIXAgent(torch.optim.Adam, local_state_dims, global_state_dims, action_dims,
                               hidden_layer_dims=hidden_layer_dims, optimiser_args={"lr": settings.QMIX_LR})

        super().__init__(env, test_env, self.agent)

        Helper.output_information("Multi Agent: " + str(multi_agent_settings.AGENT_COUNT))
        Helper.output_information("  - Training Steps: " + str(self.max_steps))

    def train(self):
        self.start_time = time.time()

        # Evaluate for the first time:
        optimal_policy_reward, optimal_policy_speed = self.test()
        self.store_optimal_policy_results(optimal_policy_reward, optimal_policy_speed,
                                          ['Team Returns', 'Average Speed'])

        Helper.output_information("\n\nBeginning Training")
        while self.steps < self.max_steps:
            done = False
            local_states, infos = self.env.reset()
            global_state = self.env.get_global_state()

            episode_reward = 0

            starting_episode_steps = self.steps
            while not done:
                actions = self.agent.get_action(local_states)
                next_local_states, team_reward, done, trunc, infos = self.env.step(actions)
                next_global_state = self.env.get_global_state()

                rewards, dones = infos["agents_rewards"], infos["agents_dones"]

                # If for training, we want to wait for all agents to be done before ending the episode, then update done
                if multi_agent_settings.WAIT_UNTIL_ALL_AGENTS_TERMINATED[0]:
                    done = all(dones)

                self.agent.store_experience_in_replay_buffer(
                    local_states, global_state, actions, rewards, next_local_states, next_global_state, dones
                )

                self.agent.learn()
                self.steps += 1

                if self.steps % settings.PLOT_STEPS_FREQUENCY == 0:
                    optimal_policy_reward, optimal_policy_speed = self.test()
                    self.store_optimal_policy_results(optimal_policy_reward, optimal_policy_speed,
                                                      ['Team Returns', 'Average Speed'])

                local_states = next_local_states
                global_state = next_global_state
                episode_reward += team_reward

            self.episode += 1

            # Output episode results
            self.output_episode_results(episode_reward, self.steps - starting_episode_steps)

    def test(self):
        done = False
        local_states, infos = self.test_env.reset()

        episode_steps = 0
        episode_reward = 0
        avg_agents_speed = np.empty(0)
        Helper.output_information("-------------")
        Helper.output_information("Testing Optimal Policy: " + str(self.steps / settings.PLOT_STEPS_FREQUENCY))
        while not done:
            actions = self.agent.get_action(local_states, training=False)
            local_states, team_reward, done, trunc, infos = self.test_env.step(actions)

            # If for testing, we want to wait for all agents to be done before ending the episode, then update done
            if multi_agent_settings.WAIT_UNTIL_ALL_AGENTS_TERMINATED[1]:
                done = all(infos["dones"])

            episode_reward += team_reward
            avg_agents_speed = np.append(avg_agents_speed, sum(infos["agents_speeds"]) / len(infos["agents_speeds"]))
            episode_steps += 1

        avg_speed = np.mean(avg_agents_speed)
        Helper.output_information(" - Steps: " + str(episode_steps))
        Helper.output_information(" - Reward: " + str(episode_reward))
        Helper.output_information(" - Agent Speed: " + str(avg_speed))
        Helper.output_information("-------------\n")

        return episode_reward, avg_speed
