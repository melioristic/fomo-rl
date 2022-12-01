from inputoutput import read_benchmark_data
from fomo_env import Fomo
from stable_baselines3 import PPO

# Read in data.
train, val, test = read_benchmark_data()

# Instantiate the env.
env = Fomo(train = train, val = val, test = test)

model = PPO.load("./model/best_model")

# Test the trained agent and save animation
obs = env.reset()

n_steps = 10000
tot_reward = 0
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    # Remember that the observation is the percentage of bin widths

    # 0:5 radiation
    print("Radiation bins: ", str(obs['obs'][0:5]))
    # 5:10 precipitation
    print("Precipitation bins: ", str(obs['obs'][5:10]))
    # 10:15 temperature
    print("Temperature bins: ", str(obs['obs'][10:15]))
    tot_reward += reward
    #print("Step {}".format(step + 1),"Action: ", action, 'Tot. Reward: %g'%(tot_reward))

    if done:
        print("Tolerance reached!", "current reward (F1) =", reward, "tot. reward=", tot_reward)
        break
