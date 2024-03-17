import warnings

from src.AgentRunners import SingleAgentRunner
from src.Agents import AgentFactory
from src.Utilities import settings
from src.Utilities.Helper import Helper

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


def run_single_agent():
    settings.RUN_TYPE = "SingleAgent"
    settings.SAVE_DIR = settings.PARENT_DIR + "/" + settings.RUN_TYPE + "/" + settings.SUB_DIR

    env = Helper.initialise_env()
    # TODO: Also record every single one of these? put in separate optimal_policy subdirectory?
    test_env = Helper.initialise_env(record_env=False)

    agent = AgentFactory.create_new_agent(env)
    single_agent_runner = SingleAgentRunner(env, test_env, agent)
    single_agent_runner.train()
    single_agent_runner.save_final_results()
    single_agent_runner.test()


if __name__ == "__main__":
    run_single_agent()
