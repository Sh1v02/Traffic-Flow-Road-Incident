import numpy as np
import torch

from src.Agents.Agent import Agent
from src.Buffers import PPOReplayBuffer
from src.Models import PPOActorNetwork, PPOCriticNetwork
from src.Utilities import settings
from src.Wrappers.GPUSupport import tensor


# lr = 0.0003, 0.001
# gamma = 0.8
# batch_size = 5, update_frequency = 20
# observation -> absolute = False

# TODO: Tensorboard
# TODO: Agent parent class (also includes static agent_specific_config to be called in the get_config_dict method)
class PPOAgent(Agent):
    def __init__(self, state_dims, action_dims, optimiser=torch.optim.Adam, loss=torch.nn.MSELoss(), num_epochs=10,
                 entropy_coefficient_decay=0.01):
        self.loss = loss
        self.num_epochs = num_epochs
        self.batch_size = settings.PPO_BATCH_SIZE
        self.update_frequency = settings.PPO_UPDATE_FREQUENCY
        self.gamma = settings.PPO_DISCOUNT_FACTOR
        self.actor_lr = settings.PPO_LR[0]
        self.critic_lr = settings.PPO_LR[-1]
        self.gae_lambda = settings.PPO_GAE_LAMBDA
        self.policy_clip_epsilon = settings.PPO_EPSILON
        self.critic_coefficient = settings.PPO_CRITIC_COEFFICIENT
        self.entropy_coefficient = settings.PPO_ENTROPY_COEFFICIENT
        self.entropy_coefficient_decay = settings.PPO_ENTROPY_COEFFICIENT_DECAY
        self.entropy_coefficient_min = settings.PPO_ENTROPY_COEFFICIENT_MIN

        self.hidden_layer_dims = settings.PPO_NETWORK_DIMS
        # TODO: Add entropy coefficient decay
        self.actor = PPOActorNetwork(optimiser, loss, state_dims, action_dims, optimiser_args={"lr": self.actor_lr},
                                     hidden_layer_dims=self.hidden_layer_dims)
        self.critic = PPOCriticNetwork(optimiser, loss, state_dims, optimiser_args={"lr": self.critic_lr},
                                       hidden_layer_dims=self.hidden_layer_dims)
        self.replay_buffer = PPOReplayBuffer()

        self.steps = 0

    def get_action(self, state, training=True):
        state = tensor(state)
        action_distribution = self.actor(state)

        if not training:
            return torch.argmax(action_distribution.probs).item()

        action = action_distribution.sample()

        probability = torch.squeeze(action_distribution.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(self.critic(state)).item()

        return action, value, probability

    def store_experience_in_replay_buffer(self, *args):
        self.replay_buffer.add_experience(*args)

    def learn(self):
        self.steps += 1
        if self.steps % self.update_frequency != 0:
            return

        for epoch in range(self.num_epochs):
            # GAE calculation, A(t) at each time step
            # Aₜˡᵢₙ = δₜ + (γλ)δₜ₊₁ + (γλ)²δₜ₊₂ + ... + (γλ)^(T-𝑡+₁)δₜ₊(T-1)
            # δₜ = rₜ₊₁ + γV(sₜ₊₁) - V(sₜ)

            states, actions, values, rewards, dones, old_probabilities, batches = (
                self.replay_buffer.sample_experience(self.batch_size))

            advantages = np.empty(0, dtype=np.float32)

            total_steps = len(rewards)

            for time_step in range(total_steps - 1):
                gamma_gae_lambda = 1
                current_advantage = 0
                for t in range(time_step, total_steps - 1):
                    td = rewards[t] + (self.gamma * values[t + 1] * (1 - int(dones[t]))) - values[t]
                    current_advantage += gamma_gae_lambda * td
                    if dones[t] == 1:
                        break
                    gamma_gae_lambda *= self.gamma * self.gae_lambda
                advantages = np.append(advantages, current_advantage)

            # TODO: Is it better to remove last element overall? --> can be done by returning batches in range (len - 1)
            # Account for last
            advantages = np.append(advantages, 0)
            # Normalise advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            advantages = tensor(advantages)
            values = tensor(values)
            states = tensor(np.array(states))
            old_probabilities = tensor(old_probabilities)
            actions = tensor(actions)

            # TODO: Vectorise this
            for batch in batches:
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_values = values[batch]
                batch_old_probabilities = old_probabilities[batch]
                batch_advantages = advantages[batch]

                action_distribution = self.actor(batch_states)
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
                new_predicted_values = torch.squeeze(self.critic(batch_states))
                critic_loss = self.loss(current_values_with_advantages, new_predicted_values)

                final_loss = actor_loss + (self.critic_coefficient * critic_loss) - (self.entropy_coefficient * entropy)
                self.update_networks(final_loss)

        self.replay_buffer.clear()
        self.entropy_coefficient = max(self.entropy_coefficient * self.entropy_coefficient_decay,
                                       self.entropy_coefficient_min)

    def update_networks(self, loss):
        self.actor.zero_grad()
        self.critic.optimiser.zero_grad()

        loss.backward()

        self.actor.optimiser.step()
        self.critic.optimiser.step()

    def get_agent_specific_config(self):
        return {
            "PPO_NETWORK_DIMS": str(self.hidden_layer_dims),
            "PPO_LR": str(settings.PPO_LR),
            "PPO_BATCH_SIZE": str(settings.PPO_BATCH_SIZE),
            "PPO_UPDATE_FREQUENCY": str(settings.PPO_UPDATE_FREQUENCY),
            "PPO_GAE_LAMBDA": str(settings.PPO_GAE_LAMBDA),
            "PPO_EPSILON": str(settings.PPO_EPSILON),
            "PPO_CRITIC_COEFFICIENT": str(settings.PPO_CRITIC_COEFFICIENT),
            "PPO_ENTROPY_COEFFICIENT": str(settings.PPO_ENTROPY_COEFFICIENT),
            "PPO_ENTROPY_COEFFICIENT_DECAY": str(settings.PPO_ENTROPY_COEFFICIENT_DECAY),
            "PPO_ENTROPY_COEFFICIENT_MIN": str(settings.PPO_ENTROPY_COEFFICIENT_MIN)
        }
