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
        
        self.train = train
        self.val = val
        self.test = test


        Xd, Xs, Y = self.train
        Xd_radia = Xd[:,:,0,0]
        Xd_preci = Xd[:,:,1,0]
        Xd_tempe = Xd[:,:,2,0]
        Ytrain = (Y >= np.percentile(Y, 90))*1
        
        ###
        self.action= np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        # Apply learnt bins to feature array.
        Xdrl, bin_percentages_ = apply_drlFeatures(Xd_radia, Xd_preci, Xd_tempe, self.action, FEATURES_DIM)
        
        # Note scaled data but all parameters are the defaults.
        # TODO change anything?
        pipe = make_pipeline(StandardScaler(), LogisticRegression())
        pipe.fit(Xdrl, Ytrain)        
        
        # Use logistic regression to predict on test dataset.
        Xd_, Xs_, Y_ = self.test
        Xd_radia_ = Xd_[:,:,0,0]
        Xd_preci_ = Xd_[:,:,1,0]
        Xd_tempe_ = Xd_[:,:,2,0]
        Ytest = (Y_ >= np.percentile(Y_, 90)) * 1
        
        # Apply learnt bins to test feature array.
        Xdrl_, bin_percentages= apply_drlFeatures(Xd_radia_, Xd_preci_, Xd_tempe_, self.action, FEATURES_DIM)
        # predict test instances
        Ypreds = pipe.predict(Xdrl_)
        
        # calculate f1
        f1 = f1_score(Ytest, Ypreds, average='weighted')

        ###
        
        self.observation = bin_percentages
        self.score = f1
        self.score_init = self.score

        # The action space
        self.action_space = gym.spaces.Box(np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]),
                                           np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))

        # The observation are all the time series? (for now)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, 15))

    def reset(self):
        # Reset to initial state.
        self.action = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.score = self.score_init

        return self._get_obs()

    def _get_obs(self):
        # return observation in the format of self.observation_space
        return self.observation

    def step(self, action):

        Xd, Xs, Y = self.train

        Xd_radia = Xd[:,:,0,0]
        Xd_preci = Xd[:,:,1,0]
        Xd_tempe = Xd[:,:,2,0]
        Ytrain = (Y >= np.percentile(Y, 90))*1
        
        ###
        
        # Apply learnt bins to feature array.
        Xdrl, bin_percentages_ = apply_drlFeatures(Xd_radia, Xd_preci, Xd_tempe, action, FEATURES_DIM)
        
        # Note scaled data but all parameters are the defaults.
        # TODO change anything?
        pipe = make_pipeline(StandardScaler(), LogisticRegression())
        pipe.fit(Xdrl, Ytrain)        
        
        # Use logistic regression to predict on test dataset.
        Xd_, Xs_, Y_ = self.test
        Xd_radia_ = Xd_[:,:,0,0]
        Xd_preci_ = Xd_[:,:,1,0]
        Xd_tempe_ = Xd_[:,:,2,0]
        Ytest = (Y_ >= np.percentile(Y_, 90)) * 1
        
        # Apply learnt bins to test feature array.
        Xdrl_, bin_percentages= apply_drlFeatures(Xd_radia_, Xd_preci_, Xd_tempe_, action, FEATURES_DIM)
        # predict test instances
        Ypreds = pipe.predict(Xdrl_)
        
        # calculate f1
        reward = f1_score(Ytest, Ypreds, average='weighted')
        ###

        score_delta = np.abs(self.score - reward)
        
        self.score = reward
                
        done = False

        # Reset criterion.
        if score_delta < TOLERANCE:
            # End of episode.
            done = True

        return self._get_obs(), reward, done, {}

    def render(self, mode='console'):
        if mode == 'console':
            TODO
        else:
            raise NotImplementedError()

    def close(self):
        pass


