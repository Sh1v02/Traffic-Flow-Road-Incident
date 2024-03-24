import warnings

from src.AgentRunners import MultiAgentRunner
from src.Agents import AgentFactory
from src.Utilities import settings, multi_agent_settings
from src.Utilities.Helper import Helper

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.core")



def run():
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

    env = Helper.initialise_env(config=config)
    test_env = Helper.initialise_env(config=config, record_env=False)

    replay_buffer = None
    if multi_agent_settings.SHARED_REPLAY_BUFFER:
        replay_buffer = AgentFactory.create_shared_replay_buffer()
    agents = [AgentFactory.create_new_agent(env, replay_buffer) for _ in range(multi_agent_settings.AGENT_COUNT)]

    multi_agent_runner = MultiAgentRunner(env, test_env, agents)
    multi_agent_runner.train()
    multi_agent_runner.save_final_results()
    multi_agent_runner.test()


if __name__ == "__main__":
    run()
    print("Multi Agent Run Ended")
