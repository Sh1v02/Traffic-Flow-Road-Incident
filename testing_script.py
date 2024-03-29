import sys
from multiprocessing import set_start_method
from multiprocessing import Process
from multiprocessing import Lock

from colab_run_single_agent import configure_single_agent_on_colab
from colab_run_multi_agent import configure_multi_agent_on_colab
from src.Utilities import settings
from src.Utilities.Helper import Helper

if __name__ == "__main__":
    set_start_method("spawn")

    # ****************** CUSTOM ARGUMENTS ***************************
    if len(sys.argv) > 1:
        settings.PPO_ENTROPY_COEFFICIENT = float(sys.argv[1])
        settings.PPO_ENTROPY_COEFFICIENT_DECAY = float(sys.argv[2])

        Helper.output_information("Testing with ENTROPY_COEFFICIENT: " + str(settings.PPO_ENTROPY_COEFFICIENT),
                                  file=sys.stderr)
        Helper.output_information("Testing with ENTROPY_DECAY: " + str(settings.PPO_ENTROPY_COEFFICIENT_DECAY),
                                  file=sys.stderr)

        # ***************************************************************

    lock = Lock()
    process = Process(target=configure_multi_agent_on_colab(), args=(lock,))
    process.start()
    process.join()
