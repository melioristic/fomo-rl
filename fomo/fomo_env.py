import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

from feature_functions import apply_drlFeatures
from feature_functions import binarize_mort

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

    def __init__(self, train, val, test, features_dim=15, threshold=90, tolerance=0.001):
        super(Fomo, self).__init__()
        
        self.train = train
        self.val = val
        self.test = test
        self.threshold = threshold
        self.features_dim = features_dim
        self.tolerance = tolerance


        Xd, Xs, Y = self.train
        Xd_radia = Xd[:,:,0,0]
        Xd_preci = Xd[:,:,1,0]
        Xd_tempe = Xd[:,:,2,0]
        Ytrain = binarize_mort(Y, self.threshold)
        
        ###
        self.action= np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        # Apply learnt bins to feature array.
        Xdrl, percentages_ = apply_drlFeatures(Xd_radia, Xd_preci, Xd_tempe, self.action, self.features_dim)
        
        # Note scaled data but all parameters are the defaults.
        # TODO change anything?
        pipe = make_pipeline(StandardScaler(), LogisticRegression())
        pipe.fit(Xdrl, Ytrain)        
        
        # Use logistic regression to predict on test dataset.
        Xd_, Xs_, Y_ = self.test
        Xd_radia_ = Xd_[:,:,0,0]
        Xd_preci_ = Xd_[:,:,1,0]
        Xd_tempe_ = Xd_[:,:,2,0]
        Ytest = binarize_mort(Y_, self.threshold)
        
        # Apply learnt bins to test feature array.
        Xdrl_, percentages = apply_drlFeatures(Xd_radia_, Xd_preci_, Xd_tempe_, self.action, self.features_dim)
        # predict test instances
        Ypreds = pipe.predict(Xdrl_)
        
        # calculate f1
        self.score = f1_score(Ytest, Ypreds, average='weighted')

        self.observation = percentages
        self.score_init = self.score.copy()
        ###

        # The action space
        # TODO would be cool if this were symmetrical, so low=-1, high=1, but couldn't get it to work
        self.action_space = gym.spaces.Box(low=0.001, high=1, shape=(15,), dtype=np.float32)

        # The observation are all the time series? (for now)
        self.observation_space = gym.spaces.Dict(
            spaces={
                "obs": gym.spaces.Box(low=0, high=1, shape=(15,), dtype=np.float32),
            })

    def reset(self):
        # Reset to initial state.
        self.action = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.score = self.score_init

        return self._get_obs()

    def _get_obs(self):
        # return observation in the format of self.observation_space
        return {"obs": self.observation}

    def step(self, action):

        Xd, Xs, Y = self.train

        Xd_radia = Xd[:,:,0,0]
        Xd_preci = Xd[:,:,1,0]
        Xd_tempe = Xd[:,:,2,0]
        Ytrain = binarize_mort(Y, self.threshold)
        
        ###
        # Apply learnt bins to feature array.
        Xdrl, percentages = apply_drlFeatures(Xd_radia, Xd_preci, Xd_tempe, action, self.features_dim)
        self.observation = percentages

        # Note scaled data but all parameters are the defaults.
        # TODO change anything?
        pipe = make_pipeline(StandardScaler(), LogisticRegression())
        pipe.fit(Xdrl, Ytrain)        
        
        # Use logistic regression to predict on test dataset.
        Xd_, Xs_, Y_ = self.test
        Xd_radia_ = Xd_[:,:,0,0]
        Xd_preci_ = Xd_[:,:,1,0]
        Xd_tempe_ = Xd_[:,:,2,0]
        Ytest = binarize_mort(Y_, self.threshold)
        
        # Apply learnt bins to test feature array.
        Xdrl_, percentages_ = apply_drlFeatures(Xd_radia_, Xd_preci_, Xd_tempe_, action, self.features_dim)
        # predict test instances
        Ypreds = pipe.predict(Xdrl_)
        
        # calculate f1
        reward = f1_score(Ytest, Ypreds, average='weighted')
        ###

        score_delta = np.abs(self.score - reward)
        
        self.score = reward

        done = False

        # Reset criterion.
        if score_delta < self.tolerance:
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


