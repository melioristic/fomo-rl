import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

from inputoutput import read_benchmark_data


def percentages_numpy(numpy_array):
    percentages = numpy_array/np.sum(numpy_array)
    return percentages

def irregular_bins(numpy_array, bin_values, cumsum = True):
    """
    Partition a numpy array given bin values that induce a distribution of irregular widths.
    """
    array_range = np.max(numpy_array)-np.min(numpy_array)
    percentages = percentages_numpy(bin_values)
    bin_limits = array_range*percentages
    if cumsum:
        bin_limits = np.cumsum(bin_limits)+np.min(numpy_array)
    return bin_limits

def sum_by_bins(numpy_array, bin_limits):
    bin_idxs = np.digitize(numpy_array, bin_limits[0:-1])
    # TODO check comments on digitize behaviour
    # https://github.com/numpy/numpy/issues/4217
    # bin_limits[0:-1] or bin_limits[1:] or bin_limits[1:-1]
    array_sums = np.bincount(bin_idxs, numpy_array.ravel())
    return array_sums


def main():
    train, val, test = read_benchmark_data()

    Xd, Xs, Y = train

    Xd_radia = Xd[:,:,0,0]
    Xd_preci = Xd[:,:,1,0]
    Xd_tempe = Xd[:,:,2,0]

    mortality = (Y >= np.percentile(Y, 90))*1

    #print(np.percentile(Y, 90))

    print(np.unique(mortality, return_counts = True))

    #print(Xd_radia)

    actions_array = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

    # Currently must be divisible by 3
    # as it's assumed to have the same per feature class
    # radiation, precipitation and temperature.
    FEATURES_DIM = 15

    num_rows, num_cols = Xd_radia.shape

    # Initialize (DRL) learnt feature array.
    Xdrl = np.zeros((num_rows, FEATURES_DIM))

    # Initial feature bin distribution.
    action_percentages = percentages_numpy(actions_array)

    for i in range(num_rows):
        # Calculate bin limits.
        # For radiation.
        bin_limits_r = irregular_bins(Xd_radia[i,:], percentages_numpy(actions_array[0:5]))
        # For precipitation.
        bin_limits_p = irregular_bins(Xd_preci[i,:], percentages_numpy(actions_array[5:10]))
        # For temperature.
        bin_limits_t = irregular_bins(Xd_tempe[i,:], percentages_numpy(actions_array[10:15]))

        # Populate learnt feature array.
        # Radiation.
        Xdrl[i,0:5] = sum_by_bins(Xd_radia[i,:], bin_limits_r)
        # Precipitation.
        Xdrl[i,5:10] = sum_by_bins(Xd_preci[i,:], bin_limits_p)
        # Temperature.
        Xdrl[i,10:15] = sum_by_bins(Xd_tempe[i,:], bin_limits_t)

    print(Xdrl[0,:])



    #X = np.hstack((Xd_radia, Xd_preci, Xd_tempe))

    #X = np.hstack((Xd_radia, Xd_preci, Xd_tempe))

    #clf = RandomForestClassifier(n_estimators=500, random_state=666, oob_score=True)
    #clf.fit(X, mortality)

    # Record the OOB error.
    #oob_error = 1 - clf.oob_score_


if __name__ == "__main__":
    main()