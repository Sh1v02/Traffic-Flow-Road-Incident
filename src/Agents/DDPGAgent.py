import torch
from src.Models import DDPGActorNetwork, DDPGCriticNetwork
from src.Buffers import UniformExperienceReplayBuffer


class DDPGAgent:
    def __init__(self, state_dims, action_dims, env, optimiser=torch.optim.Adam, loss=torch.nn.MSELoss(),
                 actor_lr=0.0001, critic_lr=0.001, gamma=0.99, tau=0.005, noise=0.1, batch_size=32,
                 action_range=None):
        self.action_dims = action_dims
        self.env = env
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.noise = noise
        self.batch_size = batch_size
        self.action_range = action_range if action_range else [-1.0, 1.0]
        self.replay_buffer = UniformExperienceReplayBuffer(state_dims, action_dims, 10000)

        self.actor = DDPGActorNetwork(optimiser, loss, state_dims, action_dims, optimiser_args={"lr": self.actor_lr})
        self.critic = DDPGCriticNetwork(optimiser, loss, state_dims + action_dims,
                                        optimiser_args={"lr": self.critic_lr})
        self.target_actor = DDPGActorNetwork(optimiser, loss, state_dims, action_dims,
                                             optimiser_args={"lr": self.actor_lr})
        self.target_critic = DDPGCriticNetwork(optimiser, loss, state_dims + action_dims,
                                               optimiser_args={"lr": self.critic_lr})

        # Hard update only for the first time
        self.update_target_network_parameters(tau=1.0)

    # Selects an action based on the actor output, and adds noise if training
    def get_action(self, state, training=True):
        action = self.actor(state)

        if not training:
            return action

        action += torch.normal(mean=0.0, std=self.noise, size=(self.action_dims,))
        action = torch.tensor(self.env.action_type.bound_available_lateral_actions(action.detach().numpy()))
        return torch.clamp(action, self.action_range[0], self.action_range[1]).tolist()

    def store_experience_in_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.add_experience([state, action, reward, next_state, done])

    def update_target_network_parameters(self, tau=None):
        self.update_network_parameters(self.actor, self.target_actor, tau=tau)
        self.update_network_parameters(self.critic, self.target_critic, tau=tau)

    # Update the network using: θ' <- τθ + (1 - τ)θ'
    # This becomes a soft update for τ < 1
    def update_network_parameters(self, network, target_network, tau=None):
        tau = tau if tau else self.tau
        # all(torch.equal(self.actor.state_dict()[key], self.target_actor.state_dict()[key]) for key in self.actor.state_dict())
        for network_param, target_network_param in zip(network.state_dict().values(),
                                                       target_network.state_dict().values()):
            target_network_param.data.copy_((tau * network_param.data) + ((1.0 - tau) * target_network_param.data))

    def learn(self):
        if self.replay_buffer.size < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample_experience(self.batch_size)

        target_actor_actions = self.target_actor(next_states)

        # Critic Loss
        target_critic_q_values = self.target_critic(next_states, target_actor_actions)
        critic_q_values = self.critic(states, actions)
        target = rewards + (self.gamma * target_critic_q_values * (1.0 - dones))

        self.critic.update(critic_q_values, target)

        # Actor Loss
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions)
        self.actor.update(actor_loss)

        self.update_target_network_parameters()
