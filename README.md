# Running this code

To run this code, ensure that the desired hyperparameters are set. These can be found
in [settings](src/Utilities/settings.py) and [multi_agent_settings](src/Utilities/settings.py). To use the optimal
parameters, set [USE_OPTIMAL_PARAMETERS](src/Utilities/settings.py?parameter=USE_OPTIMAL_PARAMETERS) = True. Ensure that
you are running a valid [AGENT_TYPE](src/Utilities/settings.py?parameter=AGENT_TYPE),
from ["DDQN", "VDN", "QMIX", "IPPO", "MAPPO"].

For where to save results, update [LOCAL_DIR](src/Utilities/settings.py?parameter=LOCAL_DIR)
and [SUB_DIR](src/Utilities/settings.py?parameter=SUB_DIR).

To render the environment, ensure
that [RENDER_ENVIRONMENT](src/Utilities/settings.py?parameter=RENDER_ENVIRONMENT) = True, and update the relevant screen
parameters in [graphics_settings](src/Utilities/graphics_settings.py)

To run a multi-agent algorithm, run [multi_agent_run](src/multi_agent_run.py). This will render the environment,
assuming you set the appropriate value to True, and results will be displayed in the console.

----------------------------------------------------------------
For any issues with packages, ensure that all dependencies in [requirements.txt](requirements.txt) are installed, and
the correct version.

Alternatively, you can run:

- `pip install -r requirements.txt`

----------------------------------------------------------------

#### The foundation environment extended in this work is taken from [Farama-Foundation/HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv). The citation can be seen below:

```bibtex
@misc{highway-env,
  author = {Leurent, Edouard},
  title = {An Environment for Autonomous Driving Decision-Making},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/eleurent/highway-env}},
}
