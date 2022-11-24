import numpy as np
import os

def create_dir(path):
    os.makedirs(path, exist_ok=True)

def binarize(arr:np.array, percentile:float):
    """Binarize an array based on a percentile

    Args:
        arr (np.array): Array to be binarized
        percentile (float): Values below the percentile is 0 and above percentile is 1

    Returns:
        _type_: _description_
    """
    return (arr >= np.percentile(arr, percentile)) * 1

def arr_to_frac(arr:np.array):
    frac = arr / np.sum(arr)
    return (frac)

def cumsum_with_0(arr:np.array):
    cumsum = np.concatenate((np.array([0]), np.cumsum(arr)))
    return cumsum

def create_index(arr:np.array, bin_length:int):
    """
    Create bin-indices of an irregular partition that sums to 1 (width of bins are fraction).
    """
    frac = arr_to_frac(arr)
    bin_fractions = cumsum_with_0(frac)
    bin_limits = np.rint(bin_fractions * bin_length).astype(int)
    return bin_limits, frac

def apply_drl_features(rad, precip, temp, actions_array, n_features):
    num_rows, num_cols = rad.shape

    # Initialize (DRL) learnt feature array.
    Xdrl = np.zeros((num_rows, n_features))

    N = int(n_features/3)

    index_rad, fraction_rad = create_index(actions_array[0:N], num_cols)
    index_precip, fraction_precip = create_index(actions_array[N:N*2], num_cols)
    index_temp, fraction_temp = create_index(actions_array[N*2:N*3], num_cols)

    for i in range(5):
        Xdrl[:, i] = np.sum(rad[:, index_rad[i]:index_rad[i+1]], axis=1)
        Xdrl[:, N+i] = np.sum(precip[:, index_precip[i]:index_precip[i+1]], axis=1)
        Xdrl[:, N*2+i] = np.sum(temp[:, index_temp[i]:index_temp[i+1]], axis=1)

    fraction = np.concatenate((fraction_rad, fraction_precip, fraction_temp)).astype(np.float32)

    return Xdrl, fraction