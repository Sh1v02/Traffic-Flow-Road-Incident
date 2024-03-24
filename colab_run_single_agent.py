from src.Utilities import settings
from src.single_agent_run import run_single_agent


def configure_single_agent_on_colab():
    settings.COLAB = True
    settings.RENDER_ENVIRONMENT = False
    settings.RUN_TYPE = "SingleAgent"
    settings.SAVE_DIR = settings.GOOGLE_DRIVE_DIR + "/" + settings.RUN_TYPE + "/" + settings.SUB_DIR

    run_single_agent()


if __name__ == "__main__":
    configure_single_agent_on_colab()
