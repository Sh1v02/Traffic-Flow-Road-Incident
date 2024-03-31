import warnings

from src.AgentRunners import MultiAgentRunner
from src.AgentRunners.QMIXAgentRunner import QMIXAgentRunner
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
    settings.configure_settings()

    env, test_env = Helper.create_environments(multi_agent_config)
    state_dims, action_dims = Helper.get_env_dims(env)

    if settings.AGENT_TYPE.lower() == "qmix":
        multi_agent_runner = QMIXAgentRunner(env, test_env, state_dims, len(env.get_global_state()), action_dims,
                                             hidden_layer_dims=settings.QMIX_NETWORK_DIMS)
        multi_agent_runner.train()
        multi_agent_runner.save_final_results()
        multi_agent_runner.test()
        return

    replay_buffer = networks = None
    if multi_agent_settings.SHARED_REPLAY_BUFFER:
        replay_buffer = AgentFactory.create_shared_replay_buffer(state_dims, action_dims)

    if multi_agent_settings.PARAMETER_SHARING[0].lower() == "full":
        networks = AgentFactory.create_fully_shared_networks(state_dims, action_dims)

    agents = [
        AgentFactory.create_new_agent(state_dims, action_dims, env, replay_buffer, networks)
        for _ in range(multi_agent_settings.AGENT_COUNT)
    ]

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
