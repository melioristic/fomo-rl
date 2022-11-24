##### The file should move out of this place

###TODO Check log dir and configure it properly 



import os
from stable_baselines3.common.env_checker import check_env
from fomo.inout import read_benchmark_data
from fomo.env import Fomo

train, val, test = read_benchmark_data()

#Logging
log_dir = "fomo_log"
os.makedirs(log_dir, exist_ok=True)

# Instantiate the env
env = Fomo(train = train, val = val, test = test)

# If the environment doesn't follow the interface, an error will be thrown
check_env(env, warn=True)