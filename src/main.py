import numpy as np


def random(file_path):
    steps, optimal_policy_rewards, optimal_policy_speeds = np.loadtxt(file_path, delimiter=',')
    rewards_every_one_thousand = optimal_policy_rewards[::60]
    steps_every_one_thousand = steps[::40]

    print(np.convolve(rewards_every_one_thousand,
                                                  np.ones(100) / 100,
                                                  mode='valid'))
    print(np.mean(rewards_every_one_thousand[-100:]))

    print(np.sum(rewards_every_one_thousand >= 80))


if __name__ == '__main__':
    random("C:/Users/shiva/OneDrive/Desktop/University/Fourth Year/Dissertation/Code/Dissertation/Random/rewards asdasd.txt")