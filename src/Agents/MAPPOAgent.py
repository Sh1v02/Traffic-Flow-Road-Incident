import numpy as np
import torch

from src.Agents import PPOAgent
from src.Buffers import SharedPPOReplayBuffer, PPOReplayBuffer
from src.Models import PPOActorNetwork, MAPPOCriticNetwork
from src.Utilities import settings, multi_agent_settings
from src.Wrappers.GPUSupport import tensor


# Uses full parameter sharing for the actors and critics.
# Each agent has its own actor, and there is a shared critic between all agents, that takes in the
class MAPPOAgent(PPOAgent):
    def __init__(self, optimiser, loss, local_state_dims, global_state_dims, action_dims):
        self.value_function_input_type = settings.MAPPO_VALUE_FUNCTION_INPUT_REPRESENTATION.lower()
        self.loss = loss
        self.num_epochs = settings.MAPPO_UPDATE_EPOCHS
        self.batch_size = settings.MAPPO_BATCH_SIZE
        self.update_frequency = settings.MAPPO_UPDATE_FREQUENCY
        self.gamma = settings.MAPPO_DISCOUNT_FACTOR
        self.actor_lr = settings.MAPPO_LR[0]
        self.critic_lr = settings.MAPPO_LR[-1]
        self.gae_lambda = settings.MAPPO_GAE_LAMBDA
        self.policy_clip_epsilon = settings.MAPPO_EPSILON
        self.critic_coefficient = settings.MAPPO_CRITIC_COEFFICIENT
        self.entropy_coefficient = settings.MAPPO_ENTROPY_COEFFICIENT
        self.entropy_coefficient_decay = settings.MAPPO_ENTROPY_COEFFICIENT_DECAY
        self.entropy_coefficient_min = settings.MAPPO_ENTROPY_COEFFICIENT_MIN

        self.hidden_layer_dims = settings.MAPPO_NETWORK_DIMS
        # Full parameter sharing, so just create one network, where each agent passes in its local state
        self.actor = PPOActorNetwork(optimiser, loss, local_state_dims, action_dims,
                                     hidden_layer_dims=settings.MAPPO_NETWORK_DIMS,
                                     optimiser_args={"lr": self.actor_lr})
        self.critic = MAPPOCriticNetwork(optimiser, loss, local_state_dims, global_state_dims,
                                         self.value_function_input_type, hidden_layer_dims=settings.MAPPO_NETWORK_DIMS,
                                         optimiser_args={"lr": self.critic_lr})

        self.replay_buffer = SharedPPOReplayBuffer(global_states_buffer_required=True) \
            if multi_agent_settings.SHARED_REPLAY_BUFFER else \
            [PPOReplayBuffer(global_states_buffer_required=True) for _ in range(multi_agent_settings.AGENT_COUNT)]

        self.steps = 0

    def store_experience_in_replay_buffer(self, *args):
        if multi_agent_settings.SHARED_REPLAY_BUFFER:
            self.replay_buffer.add_experience(*args)
            return

        # If we aren't using a shared replay buffer, add to each agent's respective buffer
        for agent_index in range(multi_agent_settings.AGENT_COUNT):
            # Add each agent's experience to its respective buffer
            self.replay_buffer[agent_index].add_experience(*[arg[agent_index] for arg in args])

    def learn(self):
        self.steps += 1
        if self.steps % self.update_frequency != 0:
            return

        if multi_agent_settings.SHARED_REPLAY_BUFFER:
            local_states, actions, values, rewards, dones, old_probabilities, global_states = (
                self.replay_buffer.get_buffer_contents())
            advantages = np.empty(0, dtype=np.float32)
            for i in range(self.replay_buffer.num_agents):
                advantages = np.concatenate(
                    (advantages, self.calculate_gae(values[i], rewards[i], dones[i])), axis=0
                )
            # Normalise advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # Perform the PPO learn step
            self._learn_step(local_states, actions, values, rewards, dones, old_probabilities, global_states,
                             advantages)
            # Clear buffer and apply entropy decay
            self.replay_buffer.clear()
            self.entropy_coefficient = max(self.entropy_coefficient * self.entropy_coefficient_decay,
                                           self.entropy_coefficient_min)
            return

        for agent_index in range(multi_agent_settings.AGENT_COUNT):
            local_states, actions, values, rewards, dones, old_probabilities, global_states = (
                self.replay_buffer[agent_index].get_buffer_contents())
            advantages = self.calculate_gae(values, rewards, dones)
            # Normalise advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            self._learn_step(local_states, actions, values, rewards, dones, old_probabilities, global_states,
                             advantages)
            self.replay_buffer[agent_index].clear()

        self.entropy_coefficient = max(self.entropy_coefficient * self.entropy_coefficient_decay,
                                       self.entropy_coefficient_min)

    def _learn_step(self, local_states, actions, values, rewards, dones, old_probabilities, global_states, advantages):
        # Convert to tensors, and take into account if they need flattening
        local_states = tensor(np.array(local_states)).flatten(start_dim=0, end_dim=1) \
            if multi_agent_settings.SHARED_REPLAY_BUFFER else tensor(np.array(local_states))
        actions = tensor(actions).flatten() if multi_agent_settings.SHARED_REPLAY_BUFFER \
            else tensor(np.array(actions))
        values = tensor(values).flatten() if multi_agent_settings.SHARED_REPLAY_BUFFER \
            else tensor(np.array(values))
        old_probabilities = tensor(old_probabilities).flatten() if multi_agent_settings.SHARED_REPLAY_BUFFER \
            else tensor(np.array(old_probabilities))
        global_states = tensor(np.array(global_states)).flatten(start_dim=0, end_dim=1) \
            if multi_agent_settings.SHARED_REPLAY_BUFFER else tensor(np.array(global_states))
        advantages = tensor(np.array(advantages))

        for epoch in range(self.num_epochs):
            batches = self.replay_buffer.sample_experience(self.batch_size) if multi_agent_settings.SHARED_REPLAY_BUFFER \
                else self.replay_buffer[0].sample_experience(self.batch_size)

            for batch in batches:
                batch_local_states = local_states[batch]
                batch_actions = actions[batch]
                batch_values = values[batch]
                batch_old_probabilities = old_probabilities[batch]
                batch_global_states = global_states[batch]
                batch_global_states_to_use = self._get_value_function_input(batch_global_states, batch_local_states)
                batch_advantages = advantages[batch]

                action_distribution = self.actor(batch_local_states)
                new_probabilities = action_distribution.log_prob(batch_actions)
                entropy = action_distribution.entropy().mean()
                probability_ratios = new_probabilities.exp() / batch_old_probabilities.exp()

                # Calculate actor loss using PPO's clip function
                unclipped = batch_advantages * probability_ratios
                clipped = torch.clamp(probability_ratios, 1 - self.policy_clip_epsilon,
                                      1 + self.policy_clip_epsilon) * batch_advantages
                actor_loss = -torch.min(unclipped, clipped).mean()

                # Calculate critic loss between the returns (taking advantage into account) and new network predictions
                current_values_with_advantages = batch_advantages + batch_values
                new_predicted_values = torch.squeeze(self.critic(batch_global_states_to_use))
                critic_loss = self.loss(current_values_with_advantages, new_predicted_values)

                final_loss = actor_loss + (self.critic_coefficient * critic_loss) - (self.entropy_coefficient * entropy)
                self.update_networks(final_loss)

    def _get_value_function_input(self, global_states, local_states):
        # Just return the global states as they were
        if self.value_function_input_type == "ep":
            return global_states

        # As defined in the MAPPO paper, concat each local observation onto the global state
        return torch.cat((global_states, local_states), dim=1)

    def get_agent_specific_config(self):
        return {
            "MAPPO_NETWORK_DIMS": str(self.hidden_layer_dims),
            "MAPPO_DISCOUNT_FACTOR/GAMMA": str(settings.MAPPO_DISCOUNT_FACTOR),
            "MAPPO_VALUE_FUNCTION_INPUT_REPRESENTATION": str(settings.MAPPO_VALUE_FUNCTION_INPUT_REPRESENTATION),
            "MAPPO_CRITIC_LOSS_FUNCTION": str(settings.MAPPO_CRITIC_LOSS_FUNCTION),
            "MAPPO_LR": str(settings.MAPPO_LR),
            "MAPPO_BATCH_SIZE": str(settings.MAPPO_BATCH_SIZE),
            "MAPPO_UPDATE_EPOCHS": str(settings.MAPPO_UPDATE_EPOCHS),
            "MAPPO_UPDATE_FREQUENCY": str(settings.MAPPO_UPDATE_FREQUENCY),
            "MAPPO_GAE_LAMBDA": str(settings.MAPPO_GAE_LAMBDA),
            "MAPPO_EPSILON": str(settings.MAPPO_EPSILON),
            "MAPPO_CRITIC_COEFFICIENT": str(settings.MAPPO_CRITIC_COEFFICIENT),
            "MAPPO_ENTROPY_COEFFICIENT": str(settings.MAPPO_ENTROPY_COEFFICIENT),
            "MAPPO_ENTROPY_COEFFICIENT_DECAY": str(settings.MAPPO_ENTROPY_COEFFICIENT_DECAY),
            "MAPPO_ENTROPY_COEFFICIENT_MIN": str(settings.MAPPO_ENTROPY_COEFFICIENT_MIN)
        }
