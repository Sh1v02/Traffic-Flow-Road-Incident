from stable_baselines3 import PPO
import gymnasium as gym

if __name__ == "__main__":
    train = False
    test = True

    env = gym.make("highway-with-obstructions-v0", render_mode='rgb_array')

    if train:
        n_cpu = 1
        batch_size = 64
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            n_steps=batch_size * 12 // n_cpu,
            batch_size=batch_size,
            n_epochs=10,
            learning_rate=5e-4,
            gamma=0.8,
            verbose=2,
            seed=0,
            tensorboard_log="../sb_results/",
        )
        model.learn(int(20000))
        model.save("../sb_results/PPO_absolute=false")

    # Load and test saved model
    if test:
        model = PPO.load("../sb_results/PPO_absolute=false")
        while True:
            done = truncated = False
            obs, info = env.reset(seed=0)
            total_reward = 0
            while not (done or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                env.render()
            print("total reward: ", total_reward)
