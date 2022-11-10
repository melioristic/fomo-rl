import numpy as np

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
    return bin_limits, percentages

def sum_by_bins(numpy_array, bin_limits):
    bin_idxs = np.digitize(numpy_array, bin_limits[0:-1])
    # TODO check comments on digitize behaviour
    # https://github.com/numpy/numpy/issues/4217
    # bin_limits[0:-1] or bin_limits[1:] or bin_limits[1:-1]
    array_sums = np.bincount(bin_idxs, numpy_array.ravel())
    return array_sums

def apply_drlFeatures(Xd_radia, Xd_preci, Xd_tempe, actions_array, n_features=15):
    # TODO can be vectorized probably? also dimensions selection are hardcoded :(
    num_rows, num_cols = Xd_radia.shape

    # Initialize (DRL) learnt feature array.
    Xdrl = np.zeros((num_rows, n_features))

    for i in range(num_rows):
        # Calculate bin limits.
        # For radiation.
        bin_limits_r, percentages_r = irregular_bins(Xd_radia[i,:], percentages_numpy(actions_array[0:5]))
        # For precipitation.
        bin_limits_p, percentages_p = irregular_bins(Xd_preci[i,:], percentages_numpy(actions_array[5:10]))
        # For temperature.
        bin_limits_t, percentages_t = irregular_bins(Xd_tempe[i,:], percentages_numpy(actions_array[10:15]))

        # Populate learnt feature array.
        # Radiation.
        Xdrl[i,0:5] = sum_by_bins(Xd_radia[i,:], bin_limits_r)
        # Precipitation.
        Xdrl[i,5:10] = sum_by_bins(Xd_preci[i,:], bin_limits_p)
        # Temperature.
        Xdrl[i,10:15] = sum_by_bins(Xd_tempe[i,:], bin_limits_t)
        
        # Concatenate bin limit arrays.
        bin_percentages = np.concatenate(percentages_r, percentages_p, percentages_t))
        
    return Xdrl, bin_percentages