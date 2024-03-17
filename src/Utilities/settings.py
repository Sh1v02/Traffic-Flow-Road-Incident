from datetime import datetime

AGENT_TYPE = "ppo"

SEED = 4
PLOT_STEPS_FREQUENCY = 25
TRAINING_STEPS = 50_000
RENDER_ENVIRONMENT = True
RECORD_EPISODES = [False, 500]
LOG_TENSORBOARD = False

DISCOUNT_FACTOR = 0.85

# --------------------------- PPO Settings ---------------------------
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

# C:\Users\shiva\OneDrive\Desktop\University\Fourth Year\Dissertation\Code\Dissertation\TESTING\SingleAgent\PPO\2_lanes\1_agent\15-03-24_16-22-13
# DISCOUNT_FACTOR/GAMMA,0.85
# PPO_NETWORKS,"[256, 512, 256]"
# PPO_LR,"[0.0005, 0.0003]"
# PPO_BATCH_SIZE,32
# PPO_UPDATE_FREQUENCY,320
# PPO_GAE_LAMBDA,0.95
# PPO_EPSILON,0.2
# PPO_CRITIC_COEFFICIENT,1
# PPO_ENTROPY_COEFFICIENT,0.1

date_as_str = datetime.now().strftime("%d-%m-%y_%H-%M-%S")

COLAB = False
google_drive = "/content/drive/My Drive/Dissertation"
PARENT_DIR = "TESTING"
SUB_DIR = AGENT_TYPE.upper() + "/2_lanes/1_agent/" + date_as_str



RUN_TYPE = ""
SAVE_DIR = ""


# TODO: Test with 0 reward for speed (which one learns to get passed the obstructions quicker)
# TODO: Test with 0 reward for the rightmost lane