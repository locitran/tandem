import numpy as np
import pandas as pd
import os
import warnings
from typing import Sequence
warnings.filterwarnings("ignore")

from ..utils.logger import LOGGER
from ..utils.settings import FEAT_STATS, dynamics_feat, structure_feat, seq_feat
from ..utils.settings import TANDEM_R20000, TANDEM_GJB2, TANDEM_RYR1, TANDEM_PKD1
from ..utils.settings import ROOT_DIR, CLUSTER
from ..features import TANDEM_FEATS

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

#### This section is written to swap cluster from validation set to training set
# Aim to fix the issue of validation loss lower than training loss
# appear in split 2


# Retrieve indices of SAVs in R20000 given a list of UniProt ID
def argSAVs(uid_list, SAVs):
    """Retrieve indices of SAVs given a list of UniProt ID
    uid_list: list of UniProt IDs
    target is a list of SAVs
    """
    return [
        i for i, sav in enumerate(SAVs) 
        if any(item in sav for item in uid_list)
    ]

# Given:
# array: target data
# indices: indices of datapoint to remove from 'target data'
# add_array: added data to 'target data'
# return new array after operation is applied and datapoints is removed from 'target data'
def removeSAVs(
    target: np.ndarray,
    indices: Sequence[int] ,
    add: np.ndarray,
):
    # data being remove from target
    removed_data = target[indices]

    # applying removal: post target 
    ptarget = np.delete(arr=target,obj=indices)
    ptarget = np.append(arr=add,values=ptarget)
    return ptarget, removed_data

def move_item(lst, src, dst):
    """Move element at index src to index dst (in-place)."""
    if not (0 <= src < len(lst)):
        raise IndexError("source index out of range")
    item = lst.pop(src)
    if dst > src:
        dst -= 1  # list shrank by 1 before this index
    if dst >= len(lst):
        lst.append(item)
    else:
        lst.insert(dst, item)

def swap_cluster(data_sorted, unid1, unid2):
    idx1 = next(
        i for i, v in enumerate(data_sorted) if unid1 in v['member_ID']
    )
    idx2 = next(
        i for i, v in enumerate(data_sorted) if unid2 in v['member_ID']
    )

    data1 = data_sorted[idx1].copy()
    data_sorted[idx1] = data_sorted[idx2]
    data_sorted[idx2] = data1
    return data_sorted

def processFolds(folds, 
    foldno,
    fromClstr,
    toClstr,                 
):
    """Swap clusters from fold i to fold j
    
    fromClstr: cluster in must not in fold {foldno}
    toClstr: cluster in must in fold {foldno}
    """
    clusters = {
        5: ['P08709', 'P00740', 'P04070', 'P22891', 'Q9Y337'], # fold 5
        6: ['P07101', 'Q8IWU9', 'P00439'], # test
        7: ['P06280', 'P17050'], # fold 1
        8: ['P13569'], # fold 2
        9: ['P08842', 'P34059', 'P15289']       , # fold 3
        10: ['P13765',  'P01911',  'P79483',  'Q30154',  'P01920',  'P04440',  'P01912',  'P04229',  'P13760',  'P20039'], # fold 4
        28: ['Q09428', 'P33527'], # fold 4
        26: ['Q04771', 'P37023', 'P36897', 'O00238'], # fold 4
        33: ['P16278'], # fold 4
    }
    foldno -= 1
    assert fromClstr in clusters, 'must in clusters'
    assert toClstr in clusters, 'must in clusters'
    
    # Process fromClster
    fromClstr_uid = clusters[fromClstr]
    fromClstr_indices = argSAVs(fromClstr_uid, folds[foldno]['train']['SAV_coords']) 

    # Process toClster
    toClstr_uid = clusters[toClstr]
    toClstr_indices = argSAVs(toClstr_uid, folds[foldno]['val']['SAV_coords']) 
    toClstr_data = folds[foldno]['val']['data'][toClstr_indices]

    assert len(fromClstr_indices) != 0, f'cluster {fromClstr} must not in fold {foldno}'
    assert len(toClstr_indices) != 0, f'cluster {toClstr} must in fold {foldno} val set'

    # remove fromClstr training set, then add toClstr_data
    ptarget, removed_data = removeSAVs(
        target=folds[foldno]['train']['data'],
        indices=fromClstr_indices,
        add=toClstr_data
    )
    folds[foldno]['train']['data'] = ptarget
    # Extract all x and y of training set
    folds[foldno]['train']['x'] = np.array([d['x'] for d in ptarget])
    folds[foldno]['train']['y'] = np.array([d['y'] for d in ptarget])
    folds[foldno]['train']['n_SAVs'] = folds[foldno]['train']['x'].shape[0]
    folds[foldno]['train']['SAV_coords'] = np.array([d['SAV'] for d in ptarget])
    folds[foldno]['train']['n_members'] = folds[foldno]['train']['n_members'] - len(clusters[fromClstr]) + len(clusters[toClstr])

    # remove toClstr from validation set, then add fromClstr (which is removed_data)
    ptarget, removed_data = removeSAVs(
        target=folds[foldno]['val']['data'],
        indices=toClstr_indices,
        add=removed_data
    )
    folds[foldno]['val']['data'] = ptarget
    # Extract all x and y of training set
    folds[foldno]['val']['x'] = np.array([d['x'] for d in ptarget])
    folds[foldno]['val']['y'] = np.array([d['y'] for d in ptarget])
    folds[foldno]['val']['n_SAVs'] = folds[foldno]['val']['x'].shape[0]
    folds[foldno]['val']['SAV_coords'] = np.array([d['SAV'] for d in ptarget])
    folds[foldno]['val']['n_members'] = folds[foldno]['val']['n_members'] - len(clusters[toClstr]) + len(clusters[fromClstr])

    return folds

def getR20000(
        feat_path=TANDEM_R20000, 
        clstr_path=CLUSTER, 
        test_percentage=0.1,
        val_percentage=0.18,
        feat_names=TANDEM_FEATS['v1.1'], 
        folder=None
    ):
    df_feat = pd.read_csv(feat_path)
    df_feat['UniProtID'] = df_feat['SAV_coords'].str.split().str[0]

    n_pathogenic, n_benign = df_feat.labels.value_counts()
    n_SAVs = n_pathogenic + n_benign

    df_clstr = pd.read_csv(clstr_path)
    df_clstr = df_clstr.drop(columns=['rep_member_length', 'member_length', 'member_similarity'], axis=1)

    _1 = ['P01891', 'P01892', 'P05534', 'P13746', 'P30443', 'P04439']
    _2 = ['P01912', 'P04229', 'P13760', 'P20039', 'P01911']
    _3 = ['P03989', 'P10319', 'P18464', 'P18465', 'P30460', 'P30464', 'P30466', 'P30475', 'P30479', 'P30480', 'P30481', 'P30484', 'P30490', 'P30491', 'P30685', 'Q04826', 'Q31610', 'P01889']
    _4 = ['P04222', 'P30504', 'P30505', 'Q29963', 'Q9TNN7', 'P10321']
    _5 = ['Q6DU44', 'P13747']
    _6 = ['Q16874', 'P04745']
    for _no in [_1, _2, _3, _4, _5, _6]:
        _rep_member = df_clstr[df_clstr['member_ID'].str.contains(_no[-1])].rep_member.values[0]
        add_member = ','.join(_no[:-1])
        add_n_member = len(_no[:-1])
        df_clstr.loc[df_clstr['rep_member'] == _rep_member, 'member_ID'] += ',' + add_member
        df_clstr.loc[df_clstr['rep_member'] == _rep_member, 'n_members'] += add_n_member
    df_clstr["member_ID"] = df_clstr["member_ID"].apply(lambda x: x.split(','))

    data = {}
    for i, row in df_clstr.iterrows():
        data[i] = {}
        clstr = row['member_ID']
        df_clstr_feats = df_feat[df_feat['UniProtID'].isin(clstr)]

        data[i]['n_SAVs'] = len(df_clstr_feats)
        data[i]['SAV_coords'] = df_clstr_feats['SAV_coords'].tolist()
        data[i]['n_members'] = row['n_members']
        data[i]['member_ID'] = clstr
        data[i]['y'] = df_clstr_feats['labels'].values
        df_clstr.at[i, 'n_SAVs'] = len(df_clstr_feats)
        df_clstr.at[i, 'member_ID'] = clstr

    ########################################################
    # Record the cluster IDs assigned to the test set
    test_cluster_IDs = []
    test_member_IDs = []
    ########################################################

    # Decendingly Sort data by no. SAVs in each cluster
    data_sorted = sorted(data.items(), key=lambda x: x[1]['n_SAVs'], reverse=True)
    data_sorted = [i[1] for i in data_sorted]

    # I want to take the element at index 63 and insert it to index 1071
    move_item(data_sorted, src=63, dst=1072)
    # data_sorted = swap_cluster(data_sorted, 'P13569', 'P08709')
    for item in data_sorted:
        item['cluster_IDs'] = data_sorted.index(item)

    for i in range(len(data_sorted)):
        if 'P29033' in data_sorted[i]['member_ID']:
            P29033 = data_sorted[i]
            LOGGER.info('> Deleting cluster P29033 from data_sorted')
            del data_sorted[i]
            n_clusters = len(data_sorted)
            LOGGER.info(f'No. clusters: {n_clusters}')
            LOGGER.info(f'No. SAVs after deleting P29033: {n_SAVs - P29033["n_SAVs"]}')
            LOGGER.info('*'*50)
            break
    test_cluster_IDs.append(P29033['cluster_IDs'])
    test_member_IDs.extend(P29033['member_ID'])

    # Split the data into 5 folds
    # Initialize folds
    n_folds, folds = 5, dict()
    for fold in range(n_folds):
        folds[fold] = dict()
        for entity in ['train', 'val', 'test']:
            folds[fold][entity] = dict()
            folds[fold][entity]['n_SAVs'], folds[fold][entity]['n_members'] = 0, 0
            folds[fold][entity]['n_clusters'], folds[fold][entity]['percent'] = 0, 0
            folds[fold][entity]['SAV_coords'], folds[fold][entity]['member_ID'] = [], []

    # LOGGER.info('> Adding cluster P29033 to the test set')
    for fold in range(n_folds):
        folds[fold]['test']['n_SAVs'] += P29033['n_SAVs']
        folds[fold]['test']['n_members'] += P29033['n_members']
        folds[fold]['test']['percent'] = folds[fold]['test']['n_SAVs'] / n_SAVs
        folds[fold]['test']['SAV_coords'].extend(P29033['SAV_coords'])
        folds[fold]['test']['member_ID'].extend(P29033['member_ID'])
        folds[fold]['test']['n_clusters'] += 1

    # Adding clusters to the test set
    test_indices = []
    i = 0
    while i < len(data_sorted):
        if i % 6 == 5:
            test_percentage = folds[0]['test']['percent']
            if test_percentage < 0.1:
                test_indices.append(i)
                test_cluster_IDs.append(data_sorted[i]['cluster_IDs'])
                test_member_IDs.extend(data_sorted[i]['member_ID'])
                for fold in range(n_folds):
                    folds[fold]['test']['n_SAVs'] += data_sorted[i]['n_SAVs']
                    folds[fold]['test']['n_members'] += data_sorted[i]['n_members']
                    folds[fold]['test']['percent'] = folds[fold]['test']['n_SAVs'] / n_SAVs
                    folds[fold]['test']['SAV_coords'].extend(data_sorted[i]['SAV_coords'])
                    folds[fold]['test']['member_ID'].extend(data_sorted[i]['member_ID'])
                    folds[fold]['test']['n_clusters'] += 1
            else:
                LOGGER.info(f'Test set percent = {test_percentage*100:.2f}% is larger than 10%: Breaking the loop')
                pass
                break
        i += 1

    LOGGER.info(f'No. adding to test set: {len(test_indices)+1}')
    LOGGER.info(f'Cluster IDs added to the test set: {test_cluster_IDs}')
    LOGGER.info(f'Member IDs added to the test set: {test_member_IDs}')

    LOGGER.info('> Delete test indices from data_sorted')
    for i in sorted(test_indices, reverse=True):
        del data_sorted[i]
    LOGGER.info(f'No. clusters after deleting test indices: {len(data_sorted)}')
    LOGGER.info(f'No. SAVs after deleting test indices: {n_SAVs - sum([data_sorted[i]["n_SAVs"] for i in test_indices])}')
    LOGGER.info('*'*50)

    i = 0
    while i < len(data_sorted):
        fold = i % 5
        current_percentage = folds[fold]['val']['percent']
        # If the current fold is less than the validation percentage
        # or if the fold is the last fold
        # Add the cluster to the validation set
        if current_percentage < val_percentage or fold == 4:
            folds[fold]['val']['n_SAVs'] += data_sorted[i]['n_SAVs']
            folds[fold]['val']['n_members'] += data_sorted[i]['n_members']
            folds[fold]['val']['percent'] = folds[fold]['val']['n_SAVs'] / n_SAVs
            folds[fold]['val']['SAV_coords'].extend(data_sorted[i]['SAV_coords'])
            folds[fold]['val']['member_ID'].extend(data_sorted[i]['member_ID'])
            folds[fold]['val']['n_clusters'] += 1

        else:
            # Adding clusters to the training set
            data_sorted.insert(i, data_sorted[i])
        i += 1


    # Adding clusters to the training set
    for fold in range(n_folds):
        for j in range(n_folds):
            if j != fold:
                folds[fold]['train']['n_SAVs'] += folds[j]['val']['n_SAVs']
                folds[fold]['train']['n_members'] += folds[j]['val']['n_members']
                folds[fold]['train']['percent'] = folds[fold]['train']['n_SAVs'] / n_SAVs
                folds[fold]['train']['SAV_coords'].extend(folds[j]['val']['SAV_coords'])
                folds[fold]['train']['member_ID'].extend(folds[j]['val']['member_ID'])
                folds[fold]['train']['n_clusters'] += folds[j]['val']['n_clusters']
    df_clstr.sort_values(by="n_SAVs", inplace=True, ascending=False, ignore_index=True)
    df_clstr['cluster_no'] = df_clstr.index + 1 

    #### Preprocess
    if folder:
        os.makedirs(folder, exist_ok=True)
        plot_label_ratio(folds, folder)
    LOGGER.info("Load R20000 dataset")
    
    SAV_coords = df_feat['SAV_coords'].values
    features = df_feat[feat_names].values
    preprocess_feat = Preprocessing(features)
    features = preprocess_feat(features)
    labels = df_feat['labels'].values
    labels = onehot_encoding(labels, 2)

    entities = ['train', 'val', 'test']
    for i, fold in folds.items():
        for entity in entities:
            data = fold[entity]
            data_SAV_coords = data['SAV_coords']
            idx = df_feat[df_feat['SAV_coords'].isin(data_SAV_coords)].index
            folds[i][entity]['x'] = features[idx]
            folds[i][entity]['y'] = labels[idx]

    for i in range(5):
        fold = folds[i]
        for _set in ['train', 'val', 'test']:
            target = fold[_set]
            data = []
            # e.g. folds[0]['train']['n_SAVs']
            for j in range(target['n_SAVs']):
                data.append(
                    {
                        'SAV': target['SAV_coords'][j],
                        'x': target['x'][j],
                        'y': target['y'][j],
                    }
                )
            target['data'] = np.array(data)

    
    ### This section is written to swap cluster from validation set to training set
    # Aim to fix the issue of validation loss lower than training loss
    # appear in split 2
    # folds = processFolds(folds, foldno=2, fromClstr=7, toClstr=8)
    # folds = processFolds(folds, foldno=1, fromClstr=8, toClstr=7)

    folds = processFolds(folds, foldno=2, fromClstr=10, toClstr=8)
    folds = processFolds(folds, foldno=4, fromClstr=8, toClstr=10)

    # folds = processFolds(folds, foldno=4, fromClstr=26, toClstr=28)
    # folds = processFolds(folds, foldno=2, fromClstr=28, toClstr=26)

    # folds = processFolds(folds, foldno=4, fromClstr=33, toClstr=28)
    # folds = processFolds(folds, foldno=2, fromClstr=28, toClstr=33)

    print_format = (
        "Fold {fold:>2} \n "
        "\tTrain n_SAVs {tr_n:,.0f} ({tr_pct:>5.3f}%)\tpath {tr_p:,.0f} ben {tr_b:,.0f} ratio {tr_r:>7.3f}\tclust {tr_c:,.0f} memb {tr_m:,.0f}\n"
        "\tVal   n_SAVs {va_n:,.0f} ({va_pct:>5.3f}%)\tpath {va_p:,.0f} ben {va_b:,.0f} ratio {va_r:>7.3f}\tclust {va_c:,.0f} memb {va_m:,.0f}\n"
        "\tTest  n_SAVs {te_n:,.0f} ({te_pct:>5.3f}%)\tpath {te_p:,.0f} ben {te_b:,.0f} ratio {te_r:>7.3f}\tclust {te_c:,.0f} memb {te_m:,.0f}"
    )
    test_percent = folds[0]['test']['n_SAVs'] / 20361
    test_n_pathogenic = folds[0]['test']['y'][:, 1].sum()
    test_n_benign = folds[0]['test']['y'][:, 0].sum()
    test_ratio = test_n_pathogenic / test_n_benign
    for fold in range(5):
        train_percent = folds[fold]['train']['n_SAVs'] / 20361
        train_n_pathogenic = folds[fold]['train']['y'][:, 1].sum()
        train_n_benign = folds[fold]['train']['y'][:, 0].sum()
        train_ratio = train_n_pathogenic / train_n_benign

        val_percent = folds[fold]['val']['n_SAVs'] / 20361
        val_n_pathogenic = folds[fold]['val']['y'][:, 1].sum()
        val_n_benign = folds[fold]['val']['y'][:, 0].sum()
        val_ratio = val_n_pathogenic / val_n_benign

        msg = print_format.format(
            fold=fold+1,
            tr_n=folds[fold]['train']['n_SAVs'], tr_pct=train_percent*100,
            tr_p=train_n_pathogenic, tr_b=train_n_benign, tr_r=train_ratio,
            tr_c=folds[fold]['train']['n_clusters'], tr_m=folds[fold]['train']['n_members'],
            va_n=folds[fold]['val']['n_SAVs'],   va_pct=val_percent*100,
            va_p=val_n_pathogenic,   va_b=val_n_benign,   va_r=val_ratio,
            va_c=folds[fold]['val']['n_clusters'],   va_m=folds[fold]['val']['n_members'],
            te_n=folds[fold]['test']['n_SAVs'],  te_pct=test_percent*100,
            te_p=test_n_pathogenic,  te_b=test_n_benign,  te_r=test_ratio,
            te_c=folds[fold]['test']['n_clusters'],  te_m=folds[fold]['test']['n_members'],
        )
        LOGGER.info(msg)

    R20000 = [SAV_coords, labels, features] 
    return folds, R20000, preprocess_feat, df_clstr

def getTestset(feat_path, feat_names, preprocess_feat, name=None):
    df = pd.read_csv(feat_path)

    LOGGER.info('*'*50)
    LOGGER.info('Missing values in the dataframe:')
    for i, feat_name in enumerate(df.columns):
        if df[feat_name].isnull().sum() > 0:
            LOGGER.info('%s: \t\t %d' % (feat_name, df[feat_name].isnull().sum()))

    SAV_coords = df['SAV_coords'].values
    features = df[feat_names].values
    features = preprocess_feat(features)
    labels = df['labels'].values # Contains NaN values

    nan_check = np.isnan(labels)
    nan_SAV_coords = SAV_coords[nan_check]
    nan_labels = labels[nan_check]
    nan_features = features[nan_check]

    notnan_SAV_coords = SAV_coords[~nan_check]
    notnan_labels = labels[~nan_check]
    notnan_labels = onehot_encoding(notnan_labels, 2)
    notnan_features = features[~nan_check]

    n_benign = np.sum(notnan_labels[:, 0])
    n_pathogenic = np.sum(notnan_labels[:, 1])
    name = name if name else 'Unknown'
    LOGGER.info(f"No. {name} SAVs {n_benign} (benign), {n_pathogenic} (pathogenic), and {nan_features.shape[0]} (NaN)")

    knw = [notnan_SAV_coords, notnan_labels, notnan_features]
    unk = [nan_SAV_coords, nan_labels, nan_features]
    return knw, unk

def np_to_dataset(x, y, shuffle=True, batch_size=32, seed=150):
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
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x), seed=seed)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

def plot_label_ratio(folds, folder):
    # Calculate the ratio of pathogenic to benign in each fold
    n_folds = len(folds)
    train_ratio = [folds[i]['train']['ratio'] for i in range(n_folds)]
    val_ratio = [folds[i]['val']['ratio'] for i in range(n_folds)]
    test_ratio = [folds[i]['test']['ratio'] for i in range(n_folds)]

    # Plot the ratios
    import matplotlib.pyplot as plt
    index = [0, 1, 2, 3, 4]
    for i in index:
        plt.bar(i+0.4, train_ratio[i], color='blue', width=0.4, edgecolor='w')
        plt.bar(i+0.2, val_ratio[i], color='red', width=0.4, edgecolor='w')
        plt.bar(i, test_ratio[i], color='g', width=0.4, edgecolor='w')
        plt.text(i+0.53, train_ratio[i], '{:.2f}'.format(train_ratio[i]), ha='center', va='bottom')
        plt.text(i+0.35, val_ratio[i], '{:.2f}'.format(val_ratio[i]), ha='center', va='bottom')
        plt.text(i, test_ratio[i], '{:.2f}'.format(test_ratio[i]), ha='center', va='bottom')
    # Create legend
    plt.bar(0, 0, color='blue', label=r'R20000$_{train}$')
    plt.bar(0, 0, color='red', label=r'R20000$_{val}$')
    plt.bar(0, 0, color='g', label=r'R20000$_{test}$')
    plt.ylabel('Pathogenic / benign ratio', fontsize=15)
    plt.xticks([0.2, 1.2, 2.2, 3.2, 4.2], ('Split 1', 'Split 2', 'Split 3', 'Split 4', 'Split 5'))
    plt.legend(loc=[0.9, 0.8], fontsize=10)
    # Remove spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.tight_layout()
    out = os.path.join(folder, 'label_ratio.png')
    plt.savefig(out, dpi=300)
    plt.close()
    LOGGER.info(f"Label ratio plot saved to {out}") # Write to log


