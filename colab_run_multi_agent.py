import warnings

from src.AgentRunners import MultiAgentRunner
from src.Agents import AgentFactory
from src.Utilities import settings, multi_agent_settings
from src.Utilities.Helper import Helper

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.core")


def run_multi_agent():
    settings.COLAB = True
    settings.RENDER_ENVIRONMENT = False
    settings.PARENT_DIR = settings.GOOGLE_DRIVE_DIR
    settings.RUN_TYPE = "MultiAgent"
    settings.SAVE_DIR = settings.PARENT_DIR + "/" + settings.RUN_TYPE + "/" + settings.SUB_DIR

    # Multi - agent environment configuration
    config = {
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

    env = Helper.initialise_env(config=config)
    test_env = Helper.initialise_env(config=config, record_env=False)

    agents = [AgentFactory.create_new_agent(env) for _ in range(multi_agent_settings.AGENT_COUNT)]

    multi_agent_runner = MultiAgentRunner(env, test_env, agents)
    multi_agent_runner.train()
    multi_agent_runner.save_final_results()
    multi_agent_runner.test()


if __name__ == "__main__":
    run_multi_agent()
    print("Multi Agent Run Ended")
