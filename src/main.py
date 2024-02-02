import time

import gymnasium as gym
from matplotlib import pyplot as plt

config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": True,
        "order": "sorted",
        "observe_intentions": True
    }
}
env = gym.make('highway-v0', render_mode='rgb_array')
env.configure(config)
env.reset()
env.render()
# print(obs)

for _ in range(5):
    env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()  # env.ac env.unwrapped.action_type.actions_indexes["IDLE"]
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        env.render()
    print("total reward: ", total_reward)

# plt.imshow(env.render())
# plt.show()
