import os

import gymnasium as gym
from colorama import Fore, Style
from gym.wrappers import RecordVideo

from src.Utilities import settings
from src.Wrappers.CustomEnvironmentWrapper import CustomEnvironmentWrapper


class Helper:
    @staticmethod
    def initialise_env(config=None, record_env=True):
        env = gym.make('highway-with-obstructions-v0', render_mode='rgb_array')
        if config:
            env.configure(config)
        env = CustomEnvironmentWrapper(env)
        env = Helper.record_wrap(env) if settings.RECORD_EPISODES[0] and record_env else env
        return env

    @staticmethod
    def record_wrap(env):
        video_folder = settings.SAVE_DIR + "/Videos"
        os.makedirs(video_folder, exist_ok=True)

        env = RecordVideo(
            env,
            video_folder,
            episode_trigger=lambda x: x % settings.RECORD_EPISODES[1] == 0,
            name_prefix="optimal-policy"
        )
        env.unwrapped.set_record_video_wrapper(env)
        return env

    @staticmethod
    def output_information(info, **kwargs):
        print(Fore.GREEN + info + Style.RESET_ALL, **kwargs)