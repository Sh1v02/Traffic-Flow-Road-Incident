from datetime import datetime

AGENT_TYPE = "ddqn"

SEED = 4
PLOT_STEPS_FREQUENCY = 25
TRAINING_STEPS = 50_000
RENDER_ENVIRONMENT = True
RECORD_EPISODES = [False, 500]
LOG_TENSORBOARD = False

DISCOUNT_FACTOR = 0.85

# TODO: Test PPO_NETWORK_DIMS
# TODO: Rightmost lane reward test (remove it?)
# --------------------------- PPO Settings ---------------------------
PPO_NETWORK_DIMS = [256, 256, 256]
PPO_DISCOUNT_FACTOR = 0.85
PPO_LR = [5e-4, 3e-4]
PPO_BATCH_SIZE = 32
PPO_UPDATE_FREQUENCY = 320
PPO_GAE_LAMBDA = 0.95
PPO_EPSILON = 0.2
PPO_CRITIC_COEFFICIENT = 1
PPO_ENTROPY_COEFFICIENT = 0.1
PPO_ENTROPY_COEFFICIENT_DECAY = 1
PPO_ENTROPY_COEFFICIENT_MIN = 0.001
date_as_str = datetime.now().strftime("%d-%m-%y_%H-%M-%S")

COLAB = False




GOOGLE_DRIVE_DIR = "/content/drive/My Drive/Dissertation/Results"
PARENT_DIR = "TESTING"
SUB_DIR = AGENT_TYPE.upper() + "/" + date_as_str



RUN_TYPE = ""
SAVE_DIR = ""
