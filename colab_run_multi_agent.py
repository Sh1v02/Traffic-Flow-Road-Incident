import warnings

from src.Utilities import settings
from src.multi_agent_run import run_multi_agent

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.core")


def configure_multi_agent_on_colab():
    settings.COLAB = True
    settings.RENDER_ENVIRONMENT = False
    settings.RUN_TYPE = "MultiAgent"
    settings.SAVE_DIR = settings.GOOGLE_DRIVE_DIR + "/" + settings.RUN_TYPE + "/" + settings.SUB_DIR

    run_multi_agent()


if __name__ == "__main__":
    configure_multi_agent_on_colab()
    print("Multi Agent Run Ended")
