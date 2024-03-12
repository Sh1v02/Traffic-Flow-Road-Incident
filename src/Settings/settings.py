from datetime import datetime

AGENT_TYPE = "ppo"

SEED = 0
TRAINING_STEPS = 100000
RECORD_EPISODES = [True, 200]

DISCOUNT_FACTOR = 0.85

# PPO Settings
PPO_DISCOUNT_FACTOR = 0.85
PPO_LR = [5e-4, 3e-4]
PPO_BATCH_SIZE = 64
PPO_UPDATE_FREQUENCY = 768
PPO_GAE_LAMBDA = 0.95
PPO_EPSILON = 0.2
PPO_CRITIC_COEFFICIENT = 0.5
PPO_ENTROPY_COEFFICIENT = 0.1

date_as_str = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
SAVE_DIR = AGENT_TYPE.upper() + "/NEW/100000steps/" + date_as_str
