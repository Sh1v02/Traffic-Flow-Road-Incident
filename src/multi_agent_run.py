import random
import warnings

from src.AgentRunners import MultiAgentRunner
from src.Agents import AgentFactory
from src.Utilities import settings, multi_agent_settings
from src.Utilities.Helper import Helper

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.core")

# Multi - agent environment configuration
multi_agent_config = {
    "controlled_vehicles": multi_agent_settings.AGENT_COUNT,
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
            # TODO: Try with False! (MAYBE CHANGE NORMALIZE TOO)
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


def run_multi_agent():
    if settings.RANDOM_SEED:
        settings.SEED = random.randint(0, 100)

    if settings.AGENT_TYPE == "ppo" and settings.PPO_PLOT_STEPS_PER_UPDATE:
        settings.PLOT_STEPS_FREQUENCY = settings.PPO_UPDATE_FREQUENCY

    env, test_env = Helper.create_environments(multi_agent_config)

    replay_buffer = None
    if multi_agent_settings.SHARED_REPLAY_BUFFER:
        replay_buffer = AgentFactory.create_shared_replay_buffer()
    agents = [AgentFactory.create_new_agent(env, replay_buffer) for _ in range(multi_agent_settings.AGENT_COUNT)]

    multi_agent_runner = MultiAgentRunner(env, test_env, agents)
    multi_agent_runner.train()
    multi_agent_runner.save_final_results()
    multi_agent_runner.test()


def configure_multi_agent_locally():
    settings.RUN_TYPE = "MultiAgent"
    settings.SAVE_DIR = settings.LOCAL_DIR + "/" + settings.RUN_TYPE + "/" + settings.SUB_DIR

    run_multi_agent()


if __name__ == "__main__":
    configure_multi_agent_locally()
    print("Multi Agent Run Ended")
