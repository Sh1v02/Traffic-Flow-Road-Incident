AGENT_COUNT = 2
WAIT_UNTIL_ALL_AGENTS_TERMINATED = [True, False]  # [train, test]
DEATH_HANDLING = "DEATH_MASKING"  # "STOP_ADDING", "DEATH_MASKING"
VALUE_FUNCTION_DEATH_MASKING = False
TEAM_SPIRIT = [False, 0.3, 0.3]  # Use Team Spirit, set last value > second value to interpolate
SHARED_REPLAY_BUFFER = True
PARAMETER_SHARING = ["NONE", "ONE_UPDATE"]  # [(NONE, FULL, PARTIAL), (EACH_UPDATE, ONE_UPDATE)]
NORMALIZE_GLOBAL_STATE = True
