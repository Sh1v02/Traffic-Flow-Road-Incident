import time

import numpy as np
import torch

from src.AgentRunners import AgentRunner
from src.Agents.MAPPOAgent import MAPPOAgent
from src.Utilities import multi_agent_settings, settings
from src.Utilities.Helper import Helper


class MAPPOAgentRunner(AgentRunner):
    def __init__(self, env, test_env, local_state_dims, global_state_dims, action_dims, optimiser=torch.optim.Adam,
                 loss=torch.nn.MSELoss()):
        if settings.MAPPO_CRITIC_LOSS_FUNCTION.lower() == "huber":
            loss = torch.nn.HuberLoss()
        self.agent = MAPPOAgent(optimiser, loss, local_state_dims, global_state_dims, action_dims)
        self.global_state_dims = global_state_dims
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

            global_states = [global_state for _ in range(multi_agent_settings.AGENT_COUNT)]

            episode_reward = 0

            starting_episode_steps = self.steps
            while not done:
                actions, values, probabilities = (), [], []
                for i in range(multi_agent_settings.AGENT_COUNT):
                    action, value, prob = self.agent.get_action(local_states[i], global_states[i])
                    actions += (action,)
                    values.append(value)
                    probabilities.append(prob)

                next_local_states, team_reward, done, trunc, infos = self.env.step(actions)

                rewards, dones = infos["agents_rewards"], infos["agents_dones"]

                # If for training, we want to wait for all agents to be done before ending the episode, then update done
                if multi_agent_settings.WAIT_UNTIL_ALL_AGENTS_TERMINATED[0]:
                    done = all(dones)

                # If using team spirit, update the rewards to use the team spirit calculation
                if multi_agent_settings.TEAM_SPIRIT[0]:
                    rewards = self.calculate_team_spirit_rewards(rewards, team_reward)

                self.agent.store_experience_in_replay_buffer(local_states, actions, values, rewards, dones,
                                                             probabilities, global_states)

                local_states = next_local_states
                global_state = self.env.get_global_state()

                # Update the global_states
                for agent_index in range(len(dones)):
                    # If value function death masking (set the global state to 0 here)
                    if multi_agent_settings.VALUE_FUNCTION_DEATH_MASKING and dones[agent_index]:
                        global_states[agent_index] = np.zeros(self.global_state_dims)
                    else:
                        global_states[agent_index] = global_state

                episode_reward += team_reward
                self.agent.learn()
                self.steps += 1
                if self.steps % settings.PLOT_STEPS_FREQUENCY == 0:
                    optimal_policy_reward, optimal_policy_speed = self.test()
                    self.store_optimal_policy_results(optimal_policy_reward, optimal_policy_speed,
                                                      ['Team Returns', 'Average Speed'])

            self.episode += 1
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
            actions = tuple(
                self.agent.get_action(local_states[i], training=False) for i in range(multi_agent_settings.AGENT_COUNT)
            )
            local_states, team_rewards, done, trunc, infos = self.test_env.step(actions)

            # If for testing, we want to wait for all agents to be done before ending the episode, then update done
            if multi_agent_settings.WAIT_UNTIL_ALL_AGENTS_TERMINATED[1]:
                done = all(infos["dones"])

            episode_reward += team_rewards
            avg_agents_speed = np.append(avg_agents_speed, sum(infos["agents_speeds"]) / len(infos["agents_speeds"]))
            episode_steps += 1

        avg_speed = np.mean(avg_agents_speed)
        Helper.output_information(" - Steps: " + str(episode_steps))
        Helper.output_information(" - Reward: " + str(episode_reward))
        Helper.output_information(" - Agent Speed: " + str(avg_speed))
        Helper.output_information("-------------\n")

        return episode_reward, avg_speed
