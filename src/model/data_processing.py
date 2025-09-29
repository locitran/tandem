from scipy import stats
import numpy as np

probs2mode_dtype = np.dtype([
    ('mode', 'i4'),
    ('decision', 'U20'),
    ('count', 'i4'),
    ('ratio', 'f4'),
    ('path_probs', 'f4'),
    ('path_probs_sem', 'f4')
])

def np2ds(x, y, shuffle=True, batch_size=32, seed=150):
    """Convert a numpy array to a tf.data dataset
    Args:
    data: numpy array
    shuffle: shuffle the dataset
    batch_size: batch size
        No. samples in each batch

    Returns:
    ds: tf.data.Dataset
    ds = (features, labels)
    """
    
    import sys
    if "tensorflow" not in sys.modules:
        import tensorflow as tf
    else:
        tf = sys.modules["tensorflow"]
        
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x), seed=seed)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

class Preprocessing:
    def __init__(self, data):
        self.data = data
        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0)

    def fill_na_mean(self, new_data):
        """Fill missing values with the mean of the column (only for numerical features)
        """
        for i in range(new_data.shape[1]): # Iterate through each column
            mask = np.isnan(new_data[:, i])  # Find the indices of NaN values
            new_data[mask, i] = self.mean[i] # Replace NaN values with the mean of the column
        return new_data

    def normalize(self, new_data):
        """Normalizes the new input data based on the mean and std of the training data
        """
        return (new_data - self.mean) / self.std
    
    def __call__(self, new_data):
        new_data = self.fill_na_mean(new_data)
        new_data = self.normalize(new_data)
        return new_data

def onehot_encoding(labels, n_classes):
    """One hot encodes the labels
    class 0 --> [1, 0]
    class 1 --> [0, 1]
    """
    labels = np.asarray(labels, dtype=int)
    onehot = np.zeros((len(labels), n_classes))
    for i, label in enumerate(labels):
        onehot[i, label] = 1
    return onehot

def probs2mode(probs):
    """
    Given an NxM numpy array, where N is the number of samples
    and M is the number of models' probabilities for class 1,
    return a structured array with:
      - mode: majority class (0 or 1)
      - decision: 'benign' or 'pathogenic'
      - count: number of models voting for the mode
      - path_prob: mean probability supporting the voted class
      - path_prob_mean: same as path_prob (kept for compatibility)
      - path_prob_sem: SEM of supporting probabilities
    """
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    N, M = probs.shape
    out = np.full(N, np.nan, dtype=probs2mode_dtype)

    # Convert probabilities to binary predictions
    preds = (probs > 0.5).astype(int)

    # Get mode and count across the whole dataset
    mode = stats.mode(preds, axis=1)
    mode_val = mode[0]
    mode_count = mode[1]
    decision = np.array([
        "pathogenic" if val == 1 else "benign" for val in mode_val
    ])
    ratio = mode_count / M

    # Broadcast mode_val to match preds shape
    mask = preds != mode_val[:, None]
    masked_probs = np.ma.array(probs, mask=mask)
    path_probs = masked_probs.mean(1)
    path_probs_sem = stats.sem(masked_probs, axis=1)

    out['mode'] = mode_val
    out['count'] = mode_count
    out['decision'] = decision
    out['ratio'] = ratio
    out['path_probs'] = path_probs
    out['path_probs_sem'] = path_probs_sem
    return out


