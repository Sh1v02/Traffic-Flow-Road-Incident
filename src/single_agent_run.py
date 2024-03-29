import random
import warnings

from src.AgentRunners import SingleAgentRunner
from src.Agents import AgentFactory
from src.Utilities import settings
from src.Utilities.Helper import Helper

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.core")


def run_single_agent():
    settings.configure_settings()

    env, test_env = Helper.create_environments()

    agent = AgentFactory.create_new_agent(env)
    single_agent_runner = SingleAgentRunner(env, test_env, agent)
    single_agent_runner.train()
    single_agent_runner.save_final_results()
    single_agent_runner.test()


def configure_single_agent_locally():
    settings.RUN_TYPE = "SingleAgent"
    settings.SAVE_DIR = settings.LOCAL_DIR + "/" + settings.RUN_TYPE + "/" + settings.SUB_DIR

    run_single_agent()


if __name__ == "__main__":
    configure_single_agent_locally()
