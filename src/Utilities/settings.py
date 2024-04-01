import random
from datetime import datetime
from random import seed

import torch
from numpy.random import seed as np_seed

from src.Utilities.Helper import Helper

ENVIRONMENT_SEED = 4
SEED = 4
RANDOM_SEED = False

AGENT_TYPE = "qmix"

PLOT_STEPS_FREQUENCY = 25  # Might be overridden when configuring, check PPO_PLOT_STEPS_PER_UPDATE
TRAINING_STEPS = 300_000
RENDER_ENVIRONMENT = True
RECORD_EPISODES = [False, 500]
LOG_TENSORBOARD = False

# -------------------------- DDQN Settings ----------------------------
DDQN_NETWORK_DIMS = [256, 256]
DDQN_DISCOUNT_FACTOR = 0.8
DDQN_LR = 5e-4
DDQN_EPSILON = 1.0
DDQN_EPSILON_DECAY = 0.99
DDQN_MIN_EPSILON = 0.01
DDQN_BATCH_SIZE = 32
DDQN_REPLAY_BUFFER_SIZE = 10000
DDQN_UPDATE_TARGET_NETWORK_FREQUENCY = 50

# TODO: Test PPO_NETWORK_DIMS
# TODO: Rightmost lane reward test (remove it?) -> if i do note that, is there really a need for the faster lanes if all
#   cars are coordinating well?
# --------------------------- PPO Settings ----------------------------
PPO_NETWORK_DIMS = [256, 256, 256, 256]
PPO_LR = [3e-4, 3e-3]
PPO_DISCOUNT_FACTOR = 0.9
PPO_GAE_LAMBDA = 0.95
PPO_EPSILON = 0.2
PPO_CRITIC_COEFFICIENT = 1
PPO_ENTROPY_COEFFICIENT = 0.2
PPO_ENTROPY_COEFFICIENT_DECAY = 0.999
PPO_ENTROPY_COEFFICIENT_MIN = 0.001
PPO_BATCH_SIZE = 512
PPO_UPDATE_FREQUENCY = 1536
PPO_PLOT_STEPS_PER_UPDATE = True


# -------------------------- QMIX Settings ----------------------------
QMIX_AGENT_NETWORK_DIMS = [256, 256]
QMIX_HYPER_NETWORK_DIMS = 256
QMIX_MIXER_NETWORK_DIMS = 32
QMIX_SOFT_UPDATE = False
QMIX_SOFT_UPDATE_TAU = 0.005
QMIX_HARD_UPDATE_NETWORKS_FREQUENCY = 1000  # TODO: try 50
QMIX_BATCH_SIZE = 32
QMIX_EPSILON = 1.0
QMIX_EPSILON_DECAY = 0.999999
QMIX_MIN_EPSILON = 0.05
QMIX_DISCOUNT_FACTOR = 0.99
QMIX_LR = 5e-4
QMIX_REPLAY_BUFFER_SIZE = 10000
QMIX_GRADIENT_CLIP = False
QMIX_USE_VDN_MIXER = False
QMIX_AGENT_NETWORKS_SHARED = True



date_as_str = datetime.now().strftime("%d-%m-%y_%H-%M-%S")

GOOGLE_DRIVE_DIR = "/content/drive/My Drive/Dissertation/Results"
LOCAL_DIR = "Results"
SUB_DIR = AGENT_TYPE.upper() + "/" + date_as_str

COLAB = False
RUN_TYPE = ""
SAVE_DIR = ""


# import random
# from datetime import datetime
# from random import seed
#
# import torch
# from numpy.random import seed as np_seed
#
# from src.Utilities.Helper import Helper
#
# ENVIRONMENT_SEED = 4
# SEED = 4
# RANDOM_SEED = False
#
# AGENT_TYPE = "qmix"
#
# PLOT_STEPS_FREQUENCY = 25  # Might be overridden when configuring, check PPO_PLOT_STEPS_PER_UPDATE
# TRAINING_STEPS = 300_000
# RENDER_ENVIRONMENT = True
# RECORD_EPISODES = [False, 500]
# LOG_TENSORBOARD = False
#
# # -------------------------- DDQN Settings ----------------------------
# DDQN_NETWORK_DIMS = [256, 256]
# DDQN_DISCOUNT_FACTOR = 0.8
# DDQN_LR = 5e-4
# DDQN_EPSILON = 1.0
# DDQN_EPSILON_DECAY = 0.99
# DDQN_MIN_EPSILON = 0.01
# DDQN_BATCH_SIZE = 32
# DDQN_REPLAY_BUFFER_SIZE = 10000
# DDQN_UPDATE_TARGET_NETWORK_FREQUENCY = 50
#
# # TODO: Test PPO_NETWORK_DIMS
# # TODO: Rightmost lane reward test (remove it?) -> if i do note that, is there really a need for the faster lanes if all
# #   cars are coordinating well?
# # --------------------------- PPO Settings ----------------------------
# PPO_NETWORK_DIMS = [256, 256, 256, 256]
# PPO_LR = [3e-4, 3e-3]
# PPO_DISCOUNT_FACTOR = 0.9
# PPO_GAE_LAMBDA = 0.95
# PPO_EPSILON = 0.2
# PPO_CRITIC_COEFFICIENT = 1
# PPO_ENTROPY_COEFFICIENT = 0.2
# PPO_ENTROPY_COEFFICIENT_DECAY = 0.999
# PPO_ENTROPY_COEFFICIENT_MIN = 0.001
# PPO_BATCH_SIZE = 512
# PPO_UPDATE_FREQUENCY = 1536
# PPO_PLOT_STEPS_PER_UPDATE = True
#
#
# # -------------------------- QMIX Settings ----------------------------
# QMIX_AGENT_NETWORK_DIMS = [256, 256]
# QMIX_HYPER_NETWORK_DIMS = 64
# QMIX_MIXER_NETWORK_DIMS = 32
# QMIX_SOFT_UPDATE = False
# QMIX_SOFT_UPDATE_TAU = 0.005
# QMIX_HARD_UPDATE_NETWORKS_FREQUENCY = 500  # TODO: try 50
# QMIX_BATCH_SIZE = 32
# QMIX_EPSILON = 1.0
# QMIX_EPSILON_DECAY = 0.999999
# QMIX_MIN_EPSILON = 0.05
# QMIX_DISCOUNT_FACTOR = 0.99
# QMIX_LR = 5e-4
# QMIX_REPLAY_BUFFER_SIZE = 10000
# QMIX_GRADIENT_CLIP = False
# QMIX_USE_VDN_MIXER = True
# QMIX_AGENT_NETWORKS_SHARED = True
#
#
#
# date_as_str = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
#
# GOOGLE_DRIVE_DIR = "/content/drive/My Drive/Dissertation/Results"
# LOCAL_DIR = "Results"
# SUB_DIR = AGENT_TYPE.upper() + "/" + date_as_str
#
# COLAB = False
# RUN_TYPE = ""
# SAVE_DIR = ""
#
#
# def configure_settings():
#     # This has to remain constant to ensure that the environment itself, such as road and car positions, doesn't change
#     np_seed(ENVIRONMENT_SEED)
#
#     global SEED
#     global PLOT_STEPS_FREQUENCY
#
#     if RANDOM_SEED:
#         SEED = random.randint(0, 100)
#
#     if AGENT_TYPE == "ppo" and PPO_PLOT_STEPS_PER_UPDATE:
#         PLOT_STEPS_FREQUENCY = PPO_UPDATE_FREQUENCY
#
#     torch.manual_seed(SEED)
#     torch.cuda.manual_seed(SEED)
#     torch.mps.manual_seed(SEED)
#     seed(SEED)
#     Helper.output_information("SEED: " + str(SEED))
#     if torch.cuda.is_available():
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False


def configure_settings():
    # This has to remain constant to ensure that the environment itself, such as road and car positions, doesn't change
    np_seed(ENVIRONMENT_SEED)

    global SEED
    global PLOT_STEPS_FREQUENCY

    if RANDOM_SEED:
        SEED = random.randint(0, 100)

    if AGENT_TYPE == "ppo" and PPO_PLOT_STEPS_PER_UPDATE:
        PLOT_STEPS_FREQUENCY = PPO_UPDATE_FREQUENCY

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.mps.manual_seed(SEED)
    seed(SEED)
    Helper.output_information("SEED: " + str(SEED))
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
