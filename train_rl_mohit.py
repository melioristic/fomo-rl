import time
import os 

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO

from fomo.inout import read_benchmark_data
from fomo.env import FOMO
from fomo.utils import create_dir

# Read Data
train, val, test = read_benchmark_data("/data/compoundx/benchmark-dataset/")

#Create directories for the logs and the models

exp_dir = "/data/compoundx/anand/fomo_rl/"
log_dir = exp_dir + "log/"
model_dir = exp_dir + "model/"

create_dir(log_dir)
create_dir(model_dir)


# Instantiate the env
env = FOMO(train = train, val = val, test = test)

# wrap it
env = Monitor(env, log_dir)

#Callback, this built-in function will periodically evaluate the model and save the best version
eval_callback = EvalCallback(env, best_model_save_path=model_dir,
                             log_path=log_dir, eval_freq=50,
                             deterministic=False, render=False)
### Train the agent.
max_total_step_num = 2e5

def learning_rate_schedule(progress_remaining):
    start_rate = 0.0001 #0.0003
    #Can do more complicated ones like below
    #stepnum = max_total_step_num*(1-progress_remaining)
    #return 0.003 * np.piecewise(stepnum, [stepnum>=0, stepnum>4e4, stepnum>2e5, stepnum>3e5], [1.0,0.5,0.25,0.125 ])
    return start_rate * progress_remaining #linearly decreasing

PPO_model_args = {
    "learning_rate": learning_rate_schedule, #decreasing learning rate #0.0003 #can be set to constant
    "gamma": 0.91, #0.99, discount factor for futurer rewards, between 0 (only immediate reward matters) and 1 (future reward equivalent to immediate),
    "verbose": 0, #change to 1 to get more info on training steps
    #"seed": 137, #fixing the random seed
    "ent_coef": 0.01, #0, entropy coefficient, to encourage exploration
    "clip_range": 0.2 #0.2, very roughly: probability of an action can not change by more than a factor 1+clip_range
}

### Model training
starttime = time.time()

model = PPO('MultiInputPolicy', env,**PPO_model_args)
#Load previous best model parameters, we start from that
if os.path.exists( f"{model_dir}best_model_exp1.zip"):
    model.set_parameters(f"{model_dir}best_model_exp1.zip")
model.learn(max_total_step_num, callback=eval_callback)
dt = time.time()-starttime

print("Calculation took %g hr %g min %g s"%(dt//3600, (dt//60)%60, dt%60) )
###