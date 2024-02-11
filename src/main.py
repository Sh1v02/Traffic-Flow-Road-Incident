import time

import gymnasium as gym
from matplotlib import pyplot as plt

from HighwayEnv.highway_env.envs import MultiAgentWrapper

# config = {
#     "observation": {
#         "type": "Kinematics",
#         "vehicles_count": 10,
#         "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
#         "features_range": {
#             "x": [-100, 100],
#             "y": [-100, 100],
#             "vx": [-20, 20],
#             "vy": [-20, 20]
#         },
#         "absolute": True,
#         "order": "sorted",
#         "observe_intentions": True
#     }
# }


# Multi-agent environment configuration
config = {
    "controlled_vehicles": 15,
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
        }
    }
}

env = gym.make('highway-with-obstructions-v0', render_mode='rgb_array')
# env.configure(config)
env.reset(seed=0)
obs, info = env.reset()
# print(obs)

# plt.imshow(env.render())
# plt.show()
for _ in range(5):
    env.reset()
    trunc = done = False
    total_reward = 0
    while not done and not trunc:
        action = env.action_space.sample()  # env.ac env.unwrapped.action_type.actions_indexes["IDLE"]
        # env.get_vehicle_by_index
        state, reward, done, trunc, info = env.step(action)
        total_reward += reward
        env.render()
    print("total reward: ", total_reward)

# plt.imshow(env.render())
# plt.show()
