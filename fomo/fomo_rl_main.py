import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

from feature_functions import apply_drlFeatures

from inputoutput import read_benchmark_data

def main():
    train, val, test = read_benchmark_data()

    Xd, Xs, Y = train

    Xd_radia = Xd[:,:,0,0]
    Xd_preci = Xd[:,:,1,0]
    Xd_tempe = Xd[:,:,2,0]

    Ytrain = (Y >= np.percentile(Y, 90))*1

    actions_array = np.array([-0.5,-0.5,1,1,1,1,1,1,1,1,1,1,1,1,1])

    # Currently must be divisible by 3
    # as it's assumed to have the same per feature class
    # radiation, precipitation and temperature.
    FEATURES_DIM = 15

    # Apply learnt bins to feature array.
    Xdrl, bin_percentages = apply_drlFeatures(Xd_radia, Xd_preci, Xd_tempe, actions_array, FEATURES_DIM)

    print(bin_percentages.shape)

    # Note scaled data but all parameters are the defaults.
    # TODO change anything?
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(Xdrl, Ytrain)

    train, val, test = read_benchmark_data()

    Xd_, Xs_, Y_ = test

    Xd_radia_ = Xd_[:,:,0,0]
    Xd_preci_ = Xd_[:,:,1,0]
    Xd_tempe_ = Xd_[:,:,2,0]

    Ytest = (Y_ >= np.percentile(Y_, 90)) * 1

    # Apply learnt bins to test feature array.
    Xdrl_, bin_percentages= apply_drlFeatures(Xd_radia_, Xd_preci_, Xd_tempe_, actions_array, FEATURES_DIM)

    # predict test instances
    Ypreds = pipe.predict(Xdrl_)

    # calculate f1
    f1 = f1_score(Ytest, Ypreds, average='weighted')

    print(f1)

if __name__ == "__main__":
    main()

