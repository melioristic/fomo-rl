import numpy as np

def binarize_mort(arr:np.array, threshold):
    return (arr >= np.percentile(arr, threshold)) * 1

def arrpercentages(arr:np.array):
    percentages = arr / np.sum(arr)
    return(percentages)

def cumpercentages(arr:np.array):
    cumsuma = np.cumsum(arr)
    bin_percentages = np.concatenate((np.array([0]), cumsuma[:-1], np.array([1])))
    return bin_percentages

def create_index(arr:np.array,length:int):
    """
    Create bin-indices of an irregular partition that sums to 1 (width of bins are percentages).
    """
    percentages = arrpercentages(arr)
    bin_percentages = cumpercentages(percentages)
    bin_limits = np.rint(bin_percentages * length).astype(int)
    return bin_limits, percentages

def apply_drlFeatures(Xd_radia, Xd_preci, Xd_tempe, actions_array, n_features):
    num_rows, num_cols = Xd_radia.shape

    # Initialize (DRL) learnt feature array.
    Xdrl = np.zeros((num_rows, n_features))

    N = int(n_features/3)

    index_radia, percentages_radia = create_index(actions_array[0:N], num_cols)
    index_preci, percentages_preci = create_index(actions_array[N:N*2], num_cols)
    index_tempe, percentages_tempe = create_index(actions_array[N*2:N*3], num_cols)

    for i in range(5):
        Xdrl[:, i] = np.sum(Xd_radia[:, index_radia[i]:index_radia[i+1]], axis=1)
        Xdrl[:, N+i] = np.sum(Xd_preci[:, index_preci[i]:index_preci[i+1]], axis=1)
        Xdrl[:, N*2+i] = np.sum(Xd_tempe[:, index_tempe[i]:index_tempe[i+1]], axis=1)

    percentages = np.concatenate((percentages_radia, percentages_preci, percentages_tempe)).astype(np.float32)

    return Xdrl, percentages