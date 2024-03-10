import os
import warnings

import torch
import gymnasium as gym
import numpy as np
from gym.wrappers import RecordVideo

# from gym.wrappers import RecordVideo

from Agents.DDPGAgent import DDPGAgent
from Agents.DDQNAgent import DDQNAgent
from src.Agents import AgentFactory
from src.Metrics import PlotGraphs
from src.Settings import settings

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.core")


#     config = {
#         "observation": {
#             "type": "Kinematics",
#             "vehicles_count": 10,
#             "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
#             "features_range": {
#                 "x": [-100, 100],
#                 "y": [-100, 100],
#                 "vx": [-20, 20],
#                 "vy": [-20, 20]
#             },
#             "absolute": True,
#             "order": "sorted",
#             "observe_intentions": True
#         }
#     }

def run():
    env = gym.make('highway-with-obstructions-v0', render_mode='rgb_array')
    # TODO: Run 3 lanes 9 obstacles again on different seeds to get a better plot
    # TODO: Change plots to say frames, so steps * 15 (for frame count)
    save_dir = 'PPO/new_attempt_lanes_obs=2_2_entropy=0.1_batch_size=32_320'
    record_eps = False
    env = record_wrap(env, 50, save_dir) if record_eps else env

    for seed in range(1):
        print("\n\n------------------")

        save_dir = save_dir + "/seed=" + str(seed)

        agent_factory = AgentFactory()
        agent = agent_factory.create_new_agent(env)

        steps_history = np.empty(0)
        reward_history = np.empty(0)
        speed_history = np.empty(0)
        steps = 0
        episode = 0
        max_steps = 20000
        while steps < max_steps:
            done = trunc = False
            state, info = env.reset(seed=seed)
            state = state.flatten()

            if settings.AGENT_TYPE != 'ppo':
                state = torch.Tensor(state)

            episode_reward = 0
            agent_speed = np.empty(0)
            starting_episode_steps = steps
            print("seed =", seed, " - Episode: ", episode)
            while not done and not trunc:
                if settings.AGENT_TYPE == 'ppo':
                    action, value, probability = agent.get_action(state)
                else:
                    action, value, probability = agent.get_action(state), None, None

                next_state, reward, done, trunc, info = env.step(action)
                next_state = next_state.flatten()

                if not record_eps:
                    env.render()

                if settings.AGENT_TYPE != 'ppo':
                    next_state = torch.Tensor(next_state)

                if settings.AGENT_TYPE == 'ppo':
                    agent.store_experience_in_replay_buffer(state, action, value, reward, done, probability)
                else:
                    agent.store_experience_in_replay_buffer(state, action, reward, next_state, done)

                agent.learn()
                state = next_state
                episode_reward += reward
                agent_speed = np.append(agent_speed, info["agents_speeds"][0])
                steps += 1
            episode += 1
            steps_history = np.append(steps_history, steps)
            reward_history = np.append(reward_history, episode_reward)
            speed_history = np.append(speed_history, np.mean(agent_speed))
            print("  - Reward: ", episode_reward)
            print("  - Total Steps: ", steps, "/", max_steps)
            print("  - Episode Steps: ", steps - starting_episode_steps)
            print("  - Max Reward: ", np.max(reward_history))
            if episode >= 100:
                print("  - Rolling Average (100 episodes): ", np.mean(reward_history[-100:]))

        graph_plotter = PlotGraphs.PlotGraphs()
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
    print("Ended")
