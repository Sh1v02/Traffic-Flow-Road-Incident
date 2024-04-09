from src.Utilities import settings, multi_agent_settings


# Sets all parameters to their optimal values
def use_optimal_parameters():
    # -------------------------- DDQN Optimal Parameters ----------------------------
    settings.DDQN_NETWORK_DIMS = [256, 256]
    settings.DDQN_DISCOUNT_FACTOR = 0.8
    settings.DDQN_LR = 5e-4
    settings.DDQN_EPSILON = 1.0
    settings.DDQN_EPSILON_DECAY = 0.99
    settings.DDQN_MIN_EPSILON = 0.01
    settings.DDQN_BATCH_SIZE = 32
    settings.DDQN_REPLAY_BUFFER_SIZE = 10000
    settings.DDQN_UPDATE_TARGET_NETWORK_FREQUENCY = 50

    # --------------------------- PPO Optimal Parameters ----------------------------
    settings.PPO_NETWORK_DIMS = [256, 256, 256, 256]
    settings.PPO_LR = [3e-4, 3e-3]
    settings.PPO_DISCOUNT_FACTOR = 0.9
    settings.PPO_GAE_LAMBDA = 0.95
    settings.PPO_EPSILON = 0.2
    settings.PPO_CRITIC_COEFFICIENT = 1
    settings.PPO_ENTROPY_COEFFICIENT = 0.2
    settings.PPO_ENTROPY_COEFFICIENT_DECAY = 0.999
    settings.PPO_ENTROPY_COEFFICIENT_MIN = 0.001
    settings.PPO_BATCH_SIZE = 512
    settings.PPO_UPDATE_FREQUENCY = 1536
    settings.PPO_PLOT_STEPS_PER_UPDATE = True

    # -------------------------- QMIX Optimal Parameters ----------------------------
    settings.QMIX_PLOT_STEPS_PER_UPDATE = True
    settings.QMIX_LEARN_PER_EPISODE = False
    settings.QMIX_AGENT_NETWORKS_SHARED = True
    settings.QMIX_AGENT_NETWORK_DIMS = [256, 256, 256, 256]
    settings.QMIX_HYPER_NETWORK_LAYERS = 2
    settings.QMIX_HYPER_NETWORK_DIMS = 64
    settings.QMIX_MIXER_NETWORK_DIMS = 32
    settings.QMIX_DISCOUNT_FACTOR = 0.99
    settings.QMIX_LR = [5e-4, 1e-4, 250000, True]
    settings.QMIX_SOFT_UPDATE = False
    settings.QMIX_SOFT_UPDATE_TAU = 0.005
    settings.QMIX_HARD_UPDATE_NETWORKS_FREQUENCY = 500
    settings.QMIX_BATCH_SIZE = 32
    settings.QMIX_EPSILON = 1.0
    settings.QMIX_EPSILON_DECAY = 0.999999
    settings.QMIX_MIN_EPSILON = 0.05
    settings.QMIX_REPLAY_BUFFER_SIZE = 10000
    settings.QMIX_GRADIENT_CLIP = False
    settings.QMIX_PER = False
    settings.QMIX_PER_ALPHA = 0.6
    settings.QMIX_PER_BETA = 0.4
    settings.QMIX_PER_EPSILON = 0.1
    if settings.AGENT_TYPE == "vdn":
        settings.QMIX_SOFT_UPDATE = False
        settings.QMIX_LR = [5e-4, 1e-4, 250000, True]

    # -------------------------- MAPPO Optimal Parameters ---------------------------
    settings.MAPPO_VALUE_FUNCTION_INPUT_REPRESENTATION = "AS"  # Environment Provided    Agent Specific
    multi_agent_settings.VALUE_FUNCTION_DEATH_MASKING = True
    settings.MAPPO_NETWORK_DIMS = [256, 256, 256, 256]
    settings.MAPPO_CRITIC_LOSS_FUNCTION = "MSE"
    settings.MAPPO_LR = [3e-4, 3e-4]
    settings.MAPPO_DISCOUNT_FACTOR = 0.8
    settings.MAPPO_GAE_LAMBDA = 0.95
    settings.MAPPO_EPSILON = 0.2
    settings.MAPPO_CRITIC_COEFFICIENT = 1
    settings.MAPPO_ENTROPY_COEFFICIENT = 0.4
    settings.MAPPO_ENTROPY_COEFFICIENT_DECAY = 0.99999
    settings.MAPPO_ENTROPY_COEFFICIENT_MIN = 0.001
    settings.MAPPO_UPDATE_EPOCHS = 10
    settings.MAPPO_UPDATE_FREQUENCY = 1536
    settings.MAPPO_BATCH_SIZE = (settings.MAPPO_UPDATE_FREQUENCY * multi_agent_settings.AGENT_COUNT) / 6
    settings.MAPPO_PLOT_STEPS_PER_UPDATE = True
