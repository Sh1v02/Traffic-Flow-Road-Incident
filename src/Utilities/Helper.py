import math
import os

import gymnasium as gym
from colorama import Fore, Style
from gym.wrappers import RecordVideo

from src.Utilities import settings
from src.Wrappers.CustomEnvironmentWrapper import CustomEnvironmentWrapper


class Helper:
    @staticmethod
    def create_environments(config=None):
        env = Helper.initialise_env(config=config, record_env=False)

        # We want to record optimal policy (no exploration) only after the latest updates, and then not til the next

        test_env_record_frequency = [i for i in range(settings.TRAINING_STEPS // settings.PLOT_STEPS_FREQUENCY)]

        test_env = Helper.initialise_env(config=config, record_frequency=test_env_record_frequency,
                                         folder_name="OptimalPolicyVideos")

        return env, test_env

    @staticmethod
    def initialise_env(config=None, record_env=True, record_frequency=None, folder_name="TrainingVideos"):
        env = gym.make('highway-with-obstructions-v0', render_mode='rgb_array')
        if config:
            env.configure(config)
        env = CustomEnvironmentWrapper(env)
        env = Helper.record_wrap(env, folder_name, record_frequency) if (
                settings.RECORD_EPISODES[0] and record_env) else env
        return env

    @staticmethod
    def record_wrap(env, folder_name, record_frequency):
        record_frequency = record_frequency if record_frequency else \
            [i for i in range(0, settings.TRAINING_STEPS, settings.RECORD_EPISODES[1])]

        video_folder = settings.SAVE_DIR + "/" + folder_name
        os.makedirs(video_folder, exist_ok=True)

        env = RecordVideo(
            env,
            video_folder,
            episode_trigger=lambda x: x in record_frequency,
            name_prefix="optimal-policy"
        )
        env.unwrapped.set_record_video_wrapper(env)
        return env

    @staticmethod
    def get_env_dims(env):
        states, _ = env.reset()

        is_multi_agent = settings.RUN_TYPE.lower() == "multiagent"
        state_dims = len(states[0]) if is_multi_agent else len(states)
        action_dims = env.action_space[0].n if is_multi_agent else env.action_space.n

        value_function_input_type = (settings.QMIX_VALUE_FUNCTION_INPUT_REPRESENTATION if (
                settings.AGENT_TYPE.lower() == "qmix") else settings.MAPPO_VALUE_FUNCTION_INPUT_REPRESENTATION).lower()
        global_state_dims = len(env.get_global_state(env.reset()[0]))
        if value_function_input_type == "as":
            global_state_dims = global_state_dims + state_dims
        else:
            global_state_dims = global_state_dims

        return state_dims, action_dims, global_state_dims


    @staticmethod
    def output_information(info, **kwargs):
        print(Fore.GREEN + info + Style.RESET_ALL, **kwargs)
