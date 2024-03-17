# import time
#
import time

import gymnasium as gym
import torch
# # import Models.ActorNetwork as ActorNetwork
# # import Models.CriticNetwork as CriticNetwork
# from src.Models import CriticNetwork, ActorNetwork
import numpy as np
from gym.wrappers import RecordVideo
from matplotlib import pyplot as plt
# from matplotlib import pyplot as plt

def test():
    while True:
        batch_size = 5
        num_of_states = 20
        memory_indexes = np.arange(num_of_states)
        np.random.shuffle(memory_indexes)
        batches = [memory_indexes[i:i + batch_size] for i in range(0, num_of_states, batch_size)]


test()
from stable_baselines3 import DQN, DDPG

# env = gym.make("highway-with-obstructions-v0", render_mode='rgb_array')
# env = RecordVideo(env, video_folder="run",
#                   episode_trigger=lambda e: True)  # record all episodes
#
# # Provide the video recorder to the wrapped environment
# # so it can send it intermediate simulation frames.
# env.unwrapped.set_record_video_wrapper(env)
#
# # Record a video as usual
# while True:
#     obs, info = env.reset()
#     done = truncated = False
#     while not (done or truncated):
#         action = env.action_space.sample()
#         obs, reward, done, truncated, info = env.step(action)
#         # env.render()
# env.close()
#
# env = gym.make("highway-with-obstructions-v0", render_mode='rgb_array')

# for _ in range(100):
#     env.reset()
#     trunc = done = False
#     total_reward = 0
#     while not done and not trunc:
#         action = env.action_space.sample()  # env.ac env.unwrapped.action_type.actions_indexes["IDLE"]
#         # env.get_vehicle_by_index
#         state, reward, done, trunc, info = env.step(action)
#         if done:
#             print("done reward: ", reward)
#         total_reward += reward
#         flattened_states = torch.tensor(state.flatten())
#         # flattened_actions = torch.stack([torch.tensor(np.array([0, 0]), dtype=torch.float32).view(-1), torch.tensor(np.array([0.4, 0.4]), dtype=torch.float32).view(-1)], dim=0)
#         # critic_forward_pass = Critic(flattened_states, flattened_actions)
#         # actor_forward_pass = Actor(flattened_state)
#         # critic_input = torch.cat([flattened_state, flattened_action])
#         env.render()
#     print("Done: ", done)
#     print("Trunc: ", trunc)
#     print("total reward: ", total_reward)
#     time.sleep(1)

# model = DQN('MlpPolicy', env,
#             policy_kwargs=dict(net_arch=[256, 256]),
#             learning_rate=5e-4,
#             buffer_size=15000,
#             learning_starts=200,
#             batch_size=32,
#             gamma=0.8,
#             train_freq=1,
#             gradient_steps=1,
#             target_update_interval=50,
#             verbose=1,
#             tensorboard_log="sb_results/")
# model.learn(int(20000))
# model.save("sb_results/new_dqn")

# Load and test saved model
# model = DQN.load("sb_results/new_dqn")
# while True:
#     done = truncated = False
#     obs, info = env.reset()
#     total_reward = 0
#     while not (done or truncated):
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, done, truncated, info = env.step(action)
#         total_reward += reward
#         env.render()
#     print("total reward: ", total_reward)

#
# from HighwayEnv.highway_env.envs import MultiAgentWrapper
#
# config = {
#     "observation": {
#         "type": "Kinematics",
#         "vehicles_count": 10,
#         "see_behind": True,
#         "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
#         "features_range": {
#             "x": [-100, 100],
#             "y": [-100, 100],
#             "vx": [-20, 20],
#             "vy": [-20, 20]
#         },
#         "absolute": True,
#         "order": "sorted",
#         # "observe_intentions": True
#     }
# }
#
# # Multi-agent environment configuration
config = {
    "controlled_vehicles": 2,
    "absolute": True,
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",
        }
    },
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
        },
        # "steering_range": [-np.pi / 100, np.pi / 100]
    }
}
#
env = gym.make('highway-with-obstructions-v0', render_mode='rgb_array')
# env.configure(config)
env.reset()
# # env.configure(config)
# env.reset(seed=0)
# obs, info = env.reset()
# print(obs)

# plt.imshow(env.render())
# plt.show()

# Critic = CriticNetwork(torch.optim.Adam, torch.nn.MSELoss, 72, optimiser_args={"lr": 0.003})
# Actor = ActorNetwork(torch.optim.Adam, torch.nn.MSELoss,70, 2, optimiser_args={"lr": 0.003})
# for _ in range(100):
#     env.reset()
#     env.render()
#     trunc = done = False
#     total_reward = 0
#     while not done and not trunc:
#         action = env.action_space.sample()  # env.ac env.unwrapped.action_type.actions_indexes["IDLE"]
#         # env.get_vehicle_by_index
#         state, reward, done, trunc, info = env.step(np.array([0, 0]))
#         total_reward += reward
#         # flattened_states = torch.tensor(state.flatten())
#         # flattened_actions = torch.stack([torch.tensor(np.array([0, 0]), dtype=torch.float32).view(-1), torch.tensor(np.array([0.4, 0.4]), dtype=torch.float32).view(-1)], dim=0)
#         # critic_forward_pass = Critic(flattened_states, flattened_actions)
#         # print("test")
#         # actor_forward_pass = Actor(flattened_state)
#         # critic_input = torch.cat([flattened_state, flattened_action])
#         env.render()
#     print("total reward: ", total_reward)
#
plt.imshow(env.render())
plt.show()
