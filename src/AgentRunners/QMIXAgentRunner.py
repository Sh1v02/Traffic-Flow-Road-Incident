import time

import numpy as np
import torch

from src.AgentRunners import AgentRunner
from src.Agents.QMIXAgent import QMIXAgent
from src.Utilities import settings, multi_agent_settings
from src.Utilities.Helper import Helper
from src.Wrappers.GPUSupport import tensor


class QMIXAgentRunner(AgentRunner):
    def __init__(self, env, test_env, local_state_dims, global_state_dims, action_dims):
        self.agent = QMIXAgent(torch.optim.Adam, local_state_dims, global_state_dims, action_dims,
                               optimiser_args={"lr": settings.QMIX_LR[0]})

        super().__init__(env, test_env, self.agent, global_state_dims=global_state_dims)

        Helper.output_information("Multi Agent: " + str(multi_agent_settings.AGENT_COUNT))
        Helper.output_information("  - Training Steps: " + str(self.max_steps))

    def train(self):
        self.start_time = time.time()

        # Evaluate for the first time:
        optimal_policy_reward, optimal_policy_speed, optimal_policy_trunc = self.test()
        self.store_optimal_policy_results(optimal_policy_reward, optimal_policy_speed, optimal_policy_trunc,
                                          ['Team Returns', 'Average Speed', 'End Reached'])

        Helper.output_information("\n\nBeginning Training")
        while self.steps < self.max_steps:
            done = False
            dones = [False for _ in range(multi_agent_settings.AGENT_COUNT)]
            local_states, infos = self.env.reset()
            global_states = [tensor(np.zeros(self.global_state_dims)) for _ in
                             range(multi_agent_settings.AGENT_COUNT)]
            next_global_states = [tensor(np.zeros(self.global_state_dims)) for _ in
                                  range(multi_agent_settings.AGENT_COUNT)]

            if self.agent_type == "qmix":
                global_states = self.update_global_states(local_states, global_states, dones)

            episode_reward = 0

            starting_episode_steps = self.steps
            while not done:
                actions = self.agent.get_action(local_states)
                next_local_states, team_reward, done, trunc, infos = self.env.step(actions)

                rewards, dones = infos["agents_rewards"], infos["agents_dones"]

                # If for training, we want to wait for all agents to be done before ending the episode, then update done
                if multi_agent_settings.WAIT_UNTIL_ALL_AGENTS_TERMINATED[0]:
                    done = all(dones)

                if self.agent_type == "qmix":
                    next_global_states = self.update_global_states(next_local_states, next_global_states, dones)

                self.agent.store_experience_in_replay_buffer(
                    local_states, global_states, actions, rewards, next_local_states, next_global_states, dones
                )

                local_states = next_local_states

                if self.agent_type == "qmix":
                    global_states = self.update_global_states(local_states, global_states, dones)

                episode_reward += team_reward
                if not settings.QMIX_LEARN_PER_EPISODE:
                    self.agent.learn()

                self.steps += 1
                if self.steps % settings.PLOT_STEPS_FREQUENCY == 0:
                    optimal_policy_reward, optimal_policy_speed, optimal_policy_trunc = self.test()
                    self.store_optimal_policy_results(optimal_policy_reward, optimal_policy_speed, optimal_policy_trunc,
                                                      ['Team Returns', 'Average Speed', 'End Reached'])

            self.episode += 1
            if settings.QMIX_LEARN_PER_EPISODE:
                self.agent.learn(self.steps)

            # Output episode results
            self.output_episode_results(episode_reward, self.steps - starting_episode_steps)

    def test(self):
        done = trunc = False
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

        return episode_reward, avg_speed, trunc
