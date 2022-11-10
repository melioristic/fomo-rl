import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

from feature_functions import apply_drlFeatures

from inputoutput import read_benchmark_data
import gym
import matplotlib.pyplot as plt

class Fomo(gym.Env):
    """
    Custom Environment for Stable Baseline 3 for the interpretable feature learning
    """
    metadata = {'render.modes': ['console']}

    ### Constants.

    # Currently must be divisible by 3
    # as it's assumed to have the same per feature class
    # radiation, precipitation and temperature.
    FEATURES_DIM = 15

    TOLERANCE = 0.001

    ### Reward.
    REWARD_ACCURACY = 0 # Current accuracy

    def __init__(self, train, val, test):
        super(Fomo, self).__init__()

        # Initialize data.
        Xd, Xs, Y = train

        self.radiation = Xd[:, :, 0, 0]
        self.precipitation = Xd[:, :, 1, 0]
        self.temperature = Xd[:, :, 2, 0]

        # Initial distribution of bins for R,P,R.
        # namely constant and of dimension 15 (5 for each)
        self.action = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

        num_rows, num_cols = self.radiation.shape

        # Initialize (DRL) learnt feature array.
        Xdrl = np.zeros((num_rows, FEATURES_DIM))

        # Initial feature bin distribution.
        # action_percentages = percentages_numpy(actions_array)

        for i in range(num_rows):
            # Calculate bin limits.
            # For radiation.
            bin_limits_r = irregular_bins(Xd_radia[i, :], percentages_numpy(actions_array[0:5]))
            # For precipitation.
            bin_limits_p = irregular_bins(Xd_preci[i, :], percentages_numpy(actions_array[5:10]))
            # For temperature.
            bin_limits_t = irregular_bins(Xd_tempe[i, :], percentages_numpy(actions_array[10:15]))

            # Populate learnt feature array.
            # Radiation.
            Xdrl[i, 0:5] = sum_by_bins(Xd_radia[i, :], bin_limits_r)
            # Precipitation.
            Xdrl[i, 5:10] = sum_by_bins(Xd_preci[i, :], bin_limits_p)
            # Temperature.
            Xdrl[i, 10:15] = sum_by_bins(Xd_tempe[i, :], bin_limits_t)

        ###

        # The action space
        self.action_space = gym.spaces.Box(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                                           np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))

        # The observation are all the time series? (for now)
        self.observation_space = gym.spaces.Dict(
            spaces={
                "binsize": gym.spaces.Box(low=0, high=1, shape=(1, 15)),
            })

    def reset(self):
        # Reset to initial state.
        self.action = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

        return self._get_obs()

    def _get_obs(self):
        # return observation in the format of self.observation_space
        TODO
        return {"Xdrl": self.Xdrl}

    def step(self, action):
        # Agents movement
        step = action


        # Initialize (DRL) learnt feature array.
        Xdrl = np.zeros((num_rows, FEATURES_DIM))

        # Initial feature bin distribution.
        # action_percentages = percentages_numpy(actions_array)

        for i in range(num_rows):
            # Calculate bin limits.
            # For radiation.
            bin_limits_r = irregular_bins(Xd_radia[i, :], percentages_numpy(step[0:5]))
            # For precipitation.
            bin_limits_p = irregular_bins(Xd_preci[i, :], percentages_numpy(step[5:10]))
            # For temperature.
            bin_limits_t = irregular_bins(Xd_tempe[i, :], percentages_numpy(step[10:15]))

            # Populate learnt feature array.
            # Radiation.
            Xdrl[i, 0:5] = sum_by_bins(Xd_radia[i, :], bin_limits_r)
            # Precipitation.
            Xdrl[i, 5:10] = sum_by_bins(Xd_preci[i, :], bin_limits_p)
            # Temperature.
            Xdrl[i, 10:15] = sum_by_bins(Xd_tempe[i, :], bin_limits_t)

        ###
        
        TODO

        done = False

        reward = 0

        # Stop when???????
        if (TODO):
            # Reward
            reward = 0
            done = True

        return self._get_obs(), reward, done, {}

    def render(self, mode='console'):
        if mode == 'console':
            TODO
        else:
            raise NotImplementedError()

    def close(self):
        pass


