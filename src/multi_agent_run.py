import os
import warnings
from datetime import time

import torch
import gymnasium as gym
import numpy as np
from gym.wrappers import RecordVideo

# from gym.wrappers import RecordVideo

from Agents.DDPGAgent import DDPGAgent
from Agents.DDQNAgent import DDQNAgent
from src.Metrics import ResultsPlotter

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.core")


# "observation": {
#                     "type": "Kinematics",
#                     "vehicles_count": 10,
#                     # "see_behind": True,
#                     "normalize": False,
#                     "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
#                     "features_range": {
#                         "x": [-100, 100],
#                         "y": [-100, 100],
#                         "vx": [-20, 20],
#                         "vy": [-20, 20]
#                     },
#                     "absolute": True,
#                     "order": "sorted",
#                     # "observe_intentions": True
#                 },
def run():
    agent_count = 5
    # Multi - agent environment configuration
    config = {
        "controlled_vehicles": agent_count,
        "absolute": True,
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",
                "vehicles_count": 10,
                # "see_behind": True,
                "normalize": False,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute": True,
                "order": "sorted",
            },
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {
                "type": "DiscreteMetaAction",
            },
            # "steering_range": [-np.pi / 100, np.pi / 100]
        }
    }

    env = gym.make('highway-with-obstructions-v0', render_mode='rgb_array')

    save_dir = 'DDQN/MultiAgent/Agents=5_Lanes=2_obs=5_steps=50000'
    record_eps = True
    env = record_wrap(env, 100, save_dir) if record_eps else env
    env.configure(config)
    env.reset()
    for seed in range(4, 5):
        print("\n\n------------------")
        save_dir = save_dir + "/seed=" + str(seed)
        states, infos = env.reset(seed=seed)
        # agent = DDPGAgent(state.size, env.action_space.shape[0], env, noise=0.8)
        agents = []
        for a in range(agent_count):
            agents.append(DDQNAgent(states[0].size, env.action_space[0].n, env, batch_size=32,
                                    update_target_network_frequency=50, lr=0.0005, gamma=0.8))

        steps_history = np.empty(0)
        reward_history = np.empty(0)
        speed_history = np.empty(0)
        steps = 0
        episode = 0
        max_steps = 50000
        while steps < max_steps:
            done = trunc = False
            states, infos = env.reset(seed=seed)
            # TODO: CustomWrapper to support this
            states = torch.Tensor(np.array([s.flatten() for s in states]))

            episode_reward = 0
            agents_speeds = np.empty((0, agent_count))
            starting_episode_steps = steps
            print("seed =", seed, " - Episode: ", episode)
            while not done and not trunc:
                actions = ()
                for i in range(agent_count):
                    actions += (agents[i].get_action(states[i]),)
                next_states, reward, done, trunc, infos = env.step(actions)
                agents_speeds = np.vstack((agents_speeds, infos["agents_speeds"]))
                if not record_eps:
                    env.render()
                next_states = torch.Tensor(np.array([s.flatten() for s in next_states]))
                for i in range(agent_count):
                    agents[i].store_experience_in_replay_buffer(states[i], actions[i], infos["agents_rewards"][i],
                                                                next_states[i], infos["agents_dones"][i])
                    agents[i].learn()
                episode_reward += reward
                states = next_states
                steps += 1
                # done = all(infos["agents_dones"])
            episode += 1
            steps_history = np.append(steps_history, steps)
            reward_history = np.append(reward_history, episode_reward)
            speed_history = np.append(speed_history, np.mean(agents_speeds))
            print("  - Reward: ", episode_reward)
            print("  - Total Steps: ", steps, "/", max_steps)
            print("  - Episode Steps: ", steps - starting_episode_steps)
            print("  - Max Reward: ", np.max(reward_history))
            if episode >= 100:
                print("  - Rolling Average (100 episodes): ", np.mean(reward_history[-100:]))

        graph_plotter = ResultsPlotter.ResultsPlotter()
        graph_plotter.plot_graph(steps_history, reward_history, "rewards", save_dir=save_dir)
        graph_plotter.plot_graph(steps_history, speed_history, "speed_history", save_dir=save_dir,
                                 labels=['Steps', 'Speed'])
        r_avg_window = 100
        if episode >= r_avg_window:
            graph_plotter.plot_graph(steps_history[r_avg_window - 1:],
                                     np.convolve(reward_history, np.ones(r_avg_window) / r_avg_window, mode='valid'),
                                     "r_avg=" + str(r_avg_window), save_dir=save_dir)
            graph_plotter.plot_graph(steps_history[r_avg_window - 1:],
                                     np.convolve(speed_history, np.ones(r_avg_window) / r_avg_window, mode='valid'),
                                     "speed_history_r_avg=" + str(r_avg_window), save_dir=save_dir,
                                     labels=['Steps', 'Speed'])
        np.savetxt(save_dir + "/rewards.txt", (steps_history, reward_history, speed_history), delimiter=',', fmt='%d')


def record_wrap(env, frequency, save_dir):
    video_folder = save_dir + "/Videos"
    os.makedirs(video_folder, exist_ok=True)

    env = RecordVideo(
        env,
        video_folder,
        episode_trigger=lambda x: x % frequency == 0,
        name_prefix="optimal-policy"
    )
    env.unwrapped.set_record_video_wrapper(env)
    return env


if __name__ == "__main__":
    run()
    print("Multi Agent Run Ended")
