import numpy as np
import torch

from src.Buffers import PPOReplayBuffer
from src.Models import PPOActorNetwork, PPOCriticNetwork

# lr = 0.0003, 0.001
# gamma = 0.8
# batch_size = 5, update_frequency = 20
# observation -> absolute = False

class PPOAgent:
    def __init__(self, state_dims, action_dims, optimiser=torch.optim.Adam, loss=torch.nn.MSELoss(),
                 gamma=0.99, lr=0.0003, gae_lambda=0.95, policy_clip_epsilon=0.2,
                 batch_size=32, update_frequency=320, num_epochs=10):
        lr = 5e-4
        gamma = 0.8
        self.entropy_coefficient = 0.1
        self.loss = loss
        self.gamma = gamma
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.policy_clip_epsilon = policy_clip_epsilon
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.num_epochs = num_epochs

        self.actor = PPOActorNetwork(optimiser, loss, state_dims, action_dims, optimiser_args={"lr": 0.0005},
                                     hidden_layer_dims=[256, 256])
        self.critic = PPOCriticNetwork(optimiser, loss, state_dims, optimiser_args={"lr": 0.0005},
                                       hidden_layer_dims=[256, 256])
        self.replay_buffer = PPOReplayBuffer()

        self.steps = 0

    def get_action(self, state, training=True):
        state = torch.Tensor(state)
        action_distribution = self.actor(state)
        action = action_distribution.sample()

        probability = torch.squeeze(action_distribution.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(self.critic(state)).item()

        return action, value, probability

    def store_experience_in_replay_buffer(self, state, action, value, reward, done, probability):
        self.replay_buffer.add_experience([state, action, value, reward, done, probability])

    def learn(self):
        self.steps += 1
        if self.steps % self.update_frequency != 0:
            return

        for epoch in range(self.num_epochs):
            # Aₜˡᵢₙ = δₜ + (γλ)δₜ₊₁ + (γλ)²δₜ₊₂ + ... + (γλ)^(T-𝑡+₁)δₜ₊(T-1)
            # δₜ = rₜ₊₁ + γV(sₜ₊₁) - V(sₜ)

            states, actions, values, rewards, dones, old_probabilities, batches = (
                self.replay_buffer.sample_experience(self.batch_size))

            advantages = np.empty(0, dtype=np.float32)

            # TODO: Use tensors to vectorise these calcs
            for time_step in range(len(rewards) - 1):
                gamma_gae_lambda = 1
                current_advantage = 0
                for t in range(time_step, len(rewards) - 1):
                    td = rewards[t] + (self.gamma * values[t + 1] * (1 - int(dones[t]))) - values[t]
                    current_advantage += gamma_gae_lambda * td
                    if dones[t] == 1:
                        break
                    gamma_gae_lambda *= self.gamma * self.gae_lambda
                advantages = np.append(advantages, current_advantage)

            # TODO: Is it better to remove last element overall?
            # Account for last
            advantages = np.append(advantages, 0)
            advantages = torch.Tensor(advantages)
            values = torch.Tensor(values)
            states = torch.Tensor(np.array(states))
            old_probabilities = torch.Tensor(old_probabilities)
            actions = torch.Tensor(actions)

            # TODO: Vectorise this
            for batch in batches:
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_values = values[batch]
                batch_old_probabilities = old_probabilities[batch]
                batch_advantages = advantages[batch]

                action_distribution = self.actor(batch_states)
                new_probabilities = action_distribution.log_prob(batch_actions)
                entropy = action_distribution.entropy()
                probability_ratios = new_probabilities.exp() / batch_old_probabilities.exp()

                unclipped = batch_advantages * probability_ratios
                clipped = torch.clamp(probability_ratios, 1 - self.policy_clip_epsilon,
                                      1 + self.policy_clip_epsilon) * batch_advantages
                actor_loss = -torch.min(unclipped, clipped).mean()

                returns = batch_advantages + batch_values
                new_predicted_values = torch.squeeze(self.critic(batch_states))
                critic_loss = self.loss(returns, new_predicted_values)

                entropy_loss = entropy.mean()
                final_loss = actor_loss + (critic_loss * 0.5) - (0.1 * entropy_loss)
                self.update_networks(final_loss)

        self.replay_buffer.clear()

    def update_networks(self, loss):
        self.actor.zero_grad()
        self.critic.optimiser.zero_grad()

        loss.backward()

        self.actor.optimiser.step()
        self.critic.optimiser.step()
