from datetime import datetime

AGENT_TYPE = "ppo"

SEED = 0
PLOT_STEPS_FREQUENCY = 25
TRAINING_STEPS = 50000
RENDER_ENVIRONMENT = False
RECORD_EPISODES = [False, 200]

DISCOUNT_FACTOR = 0.85

# PPO Settings
PPO_DISCOUNT_FACTOR = 0.85
PPO_LR = [5e-4, 3e-4]
PPO_BATCH_SIZE = 32
PPO_UPDATE_FREQUENCY = 320
PPO_GAE_LAMBDA = 0.95
PPO_EPSILON = 0.2
PPO_CRITIC_COEFFICIENT = 0.5
PPO_ENTROPY_COEFFICIENT = 0.1

date_as_str = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
SAVE_DIR = "Results/" + AGENT_TYPE.upper() + "/" + date_as_str


# TODO: Test with 0 reward for speed (which one learns to get passed the obstructions quicker)