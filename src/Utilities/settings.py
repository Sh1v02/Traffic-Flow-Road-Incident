from datetime import datetime

AGENT_TYPE = "ddqn"

SEED = 4
PLOT_STEPS_FREQUENCY = 25
TRAINING_STEPS = 50_000
RENDER_ENVIRONMENT = True
RECORD_EPISODES = [False, 500]
LOG_TENSORBOARD = False

# -------------------------- DDQN Settings --------------------------
DDQN_NETWORK_DIMS = [256, 256]
DDQN_DISCOUNT_FACTOR = 0.8
DDQN_LR = 5e-4
DDQN_EPSILON = 1.0
DDQN_EPSILON_DECAY = 0.99
DDQN_MIN_EPSILON = 0.01
DDQN_BATCH_SIZE = 32
DDQN_UPDATE_TARGET_NETWORK_FREQUENCY = 50

# TODO: Test PPO_NETWORK_DIMS
# TODO: Rightmost lane reward test (remove it?)
# --------------------------- PPO Settings ---------------------------
PPO_NETWORK_DIMS = [256, 256, 256]
PPO_LR = [5e-4, 3e-4]
PPO_DISCOUNT_FACTOR = 0.85
PPO_GAE_LAMBDA = 0.95
PPO_EPSILON = 0.2
PPO_CRITIC_COEFFICIENT = 1
PPO_ENTROPY_COEFFICIENT = 0.1
PPO_ENTROPY_COEFFICIENT_DECAY = 1
PPO_ENTROPY_COEFFICIENT_MIN = 0.001
PPO_BATCH_SIZE = 32
PPO_UPDATE_FREQUENCY = 320




date_as_str = datetime.now().strftime("%d-%m-%y_%H-%M-%S")

GOOGLE_DRIVE_DIR = "/content/drive/My Drive/Dissertation/Results"
PARENT_DIR = "TESTING"
SUB_DIR = AGENT_TYPE.upper() + "/" + date_as_str


COLAB = False
RUN_TYPE = ""
SAVE_DIR = ""
