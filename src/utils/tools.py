import numpy as np

def take_k_largest(array, k, remaining=False):
    """
    input: array in 1D, scalar k
    remaining: boolen, all indices except k largest values
    return: indices of k largest values
    """
    indices = np.argpartition(array, -k)
    if remaining:
        return indices[:-k]
    else:
        return indices[-k:]

def take_k_smallest(array, k, remaining=False):
    """
    input: array in 1D, scalar k
    remaining: boolen, all indices except k smallest values
    return: indices of k largest values
    """
    indices = np.argpartition(array, k)
    if remaining:
        return indices[k:]
    else:
        return indices[:k]

def mask_k_indices(array, indices):
    mask = np.zeros_like(array, dtype=bool)
    mask[indices] = True
    masked_array = np.ma.array(array, mask=mask)
    return masked_array
    