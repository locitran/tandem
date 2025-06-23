import numpy as np
import pandas as pd
import logging
import os
import warnings
warnings.filterwarnings("ignore")

_LOGGER = logging.getLogger(__name__)
# from data import feat_path, clstr_path

def split_data(feat_path, clstr_path,
               test_percentage=0.1,
               val_percentage=0.18
               ):
    """Split the data into 5 folds: train, validation, and test
    Args:
        feat_path (str): Path to the feature file
        clstr_path (str): Path to the cluster file

    Returns:
        folds (dict): A dictionary containing 5 folds
            folds[i] (dict): Fold i
                train (dict): Training set
                val (dict): Validation set
                test (dict): Test set
                    n_SAVs (int): Number of SAVs
                    n_pathogenic (int): Number of pathogenic SAVs
                    n_benign (int): Number of benign SAVs
                    ratio (float): Ratio of pathogenic to benign SAVs
                    n_clusters (int): Number of clusters
                    n_members (int): Number of members
                    SAV_coords (list): List of SAV coordinates
                    member_ID (list): List of member IDs
    """
    df_feat = pd.read_csv(feat_path)
    df_feat['UniProtID'] = df_feat['SAV_coords'].str.split().str[0]
    df_feat = df_feat.drop_duplicates(subset=['SAV_coords'])

    # _LOGGER.error('*'*50)
    # _LOGGER.error('Missing values in the dataframe:')
    # for i, feat_name in enumerate(df_feat.columns):
    #     if df_feat[feat_name].isnull().sum() > 0:
    #         _LOGGER.error('%s: \t\t %d' % (feat_name, df_feat[feat_name].isnull().sum()))

    # _LOGGER.error('Assigning the mean value of feature to the missing values')
    for feat_name in df_feat.columns:
        if df_feat[feat_name].isnull().sum() > 0:
            mean_val = df_feat[feat_name].mean()
            df_feat[feat_name] = df_feat[feat_name].fillna(mean_val)
    #         _LOGGER.error('Assigning mean value to %s: %.2f' % (feat_name, mean_val))
    # _LOGGER.error('*'*50)

    n_UniProtID = len(df_feat['UniProtID'].unique())
    n_pathogenic, n_benign = df_feat.labels.value_counts()
    n_SAVs = n_pathogenic + n_benign

    df_clstr = pd.read_csv(clstr_path)
    df_clstr = df_clstr.drop(columns=['rep_member_length', 'member_length', 'member_similarity'], axis=1)
    n_clstr = len(df_clstr)
        
    _1 = ['P01891', 'P01892', 'P05534', 'P13746', 'P30443', 'P04439']
    _2 = ['P01912', 'P04229', 'P13760', 'P20039', 'P01911']
    _3 = ['P03989', 'P10319', 'P18464', 'P18465', 'P30460', 'P30464', 'P30466', 'P30475', 'P30479', 'P30480', 'P30481', 'P30484', 'P30490', 'P30491', 'P30685', 'Q04826', 'Q31610', 'P01889']
    _4 = ['P04222', 'P30504', 'P30505', 'Q29963', 'Q9TNN7', 'P10321']
    _5 = ['Q6DU44', 'P13747']
    # _6 = ['Q16874', 'P04745']
    # for _no in [_1, _2, _3, _4, _5, _6]:
    for _no in [_1, _2, _3, _4, _5]:
        _rep_member = df_clstr[df_clstr['member_ID'].str.contains(_no[-1])].rep_member.values[0]
        add_member = ','.join(_no[:-1])
        add_n_member = len(_no[:-1])
        df_clstr.loc[df_clstr['rep_member'] == _rep_member, 'member_ID'] += ',' + add_member
        df_clstr.loc[df_clstr['rep_member'] == _rep_member, 'n_members'] += add_n_member

    # for n_members, count in df_clstr.n_members.value_counts().items():
    #     _LOGGER.error('%d members: %d clusters' % (n_members, count))
    # _LOGGER.error('No. UniProtID: %d, No. SAVs: %d, and No. clusters: %d' % (n_UniProtID, n_SAVs, n_clstr))
    # _LOGGER.error('No. pathogenic SAVs: %d and benign SAVs: %d' % (n_pathogenic, n_benign))
    # _LOGGER.error('*'*50)

    data = {}
    for i, row in df_clstr.iterrows():
        data[i] = {}
        clstr = row['member_ID'].split(',')
        df_clstr_feats = df_feat[df_feat['UniProtID'].isin(clstr)]

        data[i]['n_SAVs'] = len(df_clstr_feats)
        data[i]['SAV_coords'] = df_clstr_feats['SAV_coords'].tolist()
        data[i]['n_members'] = row['n_members']
        data[i]['member_ID'] = clstr
        data[i]['y'] = df_clstr_feats['labels'].values
        # data[i]['cluster_IDs'] = i

    ########################################################
    # Record the cluster IDs assigned to the test set
    test_cluster_IDs = []
    test_member_IDs = []
    ########################################################

    # Decendingly Sort data by no. SAVs in each cluster
    data_sorted = sorted(data.items(), key=lambda x: x[1]['n_SAVs'], reverse=True)
    data_sorted = [i[1] for i in data_sorted]
    for item in data_sorted:
        item['cluster_IDs'] = data_sorted.index(item)

    for i in range(len(data_sorted)):
        if 'P29033' in data_sorted[i]['member_ID']:
            P29033 = data_sorted[i]
            # _LOGGER.error('> Deleting cluster P29033 from data_sorted')
            del data_sorted[i]
            n_clusters = len(data_sorted)
            # _LOGGER.error('No. clusters: %d', n_clusters)
            # _LOGGER.error('No. SAVs after deleting P29033: %d', n_SAVs - P29033['n_SAVs'])
            # _LOGGER.error('*'*50)
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
            folds[fold][entity]['y'] = []
            folds[fold][entity]['ratio'] = 0
            folds[fold][entity]['n_pathogenic'], folds[fold][entity]['n_benign'] = 0, 0

    # _LOGGER.error('> Adding cluster P29033 to the test set')
    for fold in range(n_folds):
        folds[fold]['test']['n_SAVs'] += P29033['n_SAVs']
        folds[fold]['test']['n_members'] += P29033['n_members']
        folds[fold]['test']['percent'] = folds[fold]['test']['n_SAVs'] / n_SAVs
        folds[fold]['test']['SAV_coords'].extend(P29033['SAV_coords'])
        folds[fold]['test']['member_ID'].extend(P29033['member_ID'])
        folds[fold]['test']['n_clusters'] += 1
        folds[fold]['test']['y'].extend(P29033['y'])

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
                    folds[fold]['test']['y'].extend(data_sorted[i]['y'])
            else:
                # _LOGGER.error('Test set percent = %.2f%% is larger than 10%%: Breaking the loop' % (test_percentage*100))
                pass
                break
        i += 1

    # _LOGGER.error('No. adding to test set: %d' % (len(test_indices)+1))
    # _LOGGER.error('Cluster IDs added to the test set: %s' % test_cluster_IDs)
    # _LOGGER.error('Member IDs added to the test set: %s' % test_member_IDs)

    # _LOGGER.error('> Delete test indices from data_sorted')
    for i in sorted(test_indices, reverse=True):
        del data_sorted[i]
    # _LOGGER.error('No. clusters after deleting test indices: %d' % len(data_sorted))
    # _LOGGER.error('No. SAVs after deleting test indices: %d' % (n_SAVs - sum([data_sorted[i]['n_SAVs'] for i in test_indices])))
    # _LOGGER.error('*'*50)

    ####################
    # Adding clusters to the training set 
    # if available
    # P55735 172 S L,5A9Q 6 172 S,285,0
    # P57740 447 D N,5A9Q 4 447 D,660,1
    # P57740 539 R I,5A9Q 4 539 R,660,0
    # Q12769 348 Y F,5A9Q 1 348 Y,1009,0
    # Q12769 351 T A,5A9Q 1 351 T,1009,0
    # Q9BW27 275 A T,5A9Q 8 275 A,434,0
    # We add Q16874 because it is in the same cluster as P04745
    # 'P55735', 'P57740', 'Q12769', 'Q9BW27' are clusters with single member
    # We add it along with P04745 
    add_to_train = []
    _Q16874_SAVs = df_feat[df_feat['UniProtID'].isin(['Q16874', 'P55735', 'P57740', 'Q12769', 'Q9BW27'])]
    # if len(_Q16874_SAVs) > 0:
    #     _LOGGER.error('> Adding cluster Q16874 P55735 P57740 Q12769 Q9BW27 to the training set')
    #     _LOGGER.error('Q16874 SAVs: %d' % len(_Q16874_SAVs))
    #     _LOGGER.error('Q16874 SAV coords: %s' % _Q16874_SAVs['SAV_coords'].values[0])
    #     _LOGGER.error('Q16874 SAV labels: %d' % _Q16874_SAVs['labels'].values[0])
    #     _LOGGER.error('*'*50)
    ####################
    i = 0
    while i < len(data_sorted):
        fold = i % 5
        current_percentage = folds[fold]['val']['percent']
        # If the current fold is less than the validation percentage
        # or if the fold is the last fold
        # Add the cluster to the validation set
        if current_percentage < val_percentage or fold == 4:
            # # _6 = ['Q16874', 'P04745']
            if 'P04745' in data_sorted[i]['member_ID'] :
                # _LOGGER.error('> Adding cluster P04745 to the validation set')
                folds[fold]['val']['n_SAVs'] += len(_Q16874_SAVs)
                folds[fold]['val']['n_members'] += 1
                folds[fold]['val']['SAV_coords'].extend(_Q16874_SAVs['SAV_coords'].values)
                folds[fold]['val']['member_ID'].extend(["Q16874"])
                folds[fold]['val']['y'].extend(_Q16874_SAVs['labels'].values)

            folds[fold]['val']['n_SAVs'] += data_sorted[i]['n_SAVs']
            folds[fold]['val']['n_members'] += data_sorted[i]['n_members']
            folds[fold]['val']['percent'] = folds[fold]['val']['n_SAVs'] / n_SAVs
            folds[fold]['val']['SAV_coords'].extend(data_sorted[i]['SAV_coords'])
            folds[fold]['val']['member_ID'].extend(data_sorted[i]['member_ID'])
            folds[fold]['val']['n_clusters'] += 1
            # folds[fold]['val']['y'].append(data_sorted[i]['y'])
            folds[fold]['val']['y'].extend(data_sorted[i]['y'])
            folds[fold]['val']['n_pathogenic'] += np.sum(data_sorted[i]['y'] == 1)
            folds[fold]['val']['n_benign'] += np.sum(data_sorted[i]['y'] == 0)
            folds[fold]['val']['ratio'] = folds[fold]['val']['n_pathogenic'] / folds[fold]['val']['n_benign']

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
                folds[fold]['train']['y'].extend(folds[j]['val']['y'])

    # Concatenate the features and labels
    for fold in range(n_folds):
        for entity in ['train', 'val', 'test']:
            # folds[fold][entity]['y'] = np.concatenate(folds[fold][entity]['y'], axis=0)
            folds[fold][entity]['y'] = np.array(folds[fold][entity]['y'])
            folds[fold][entity]['n_pathogenic'] = np.sum(folds[fold][entity]['y'] == 1)
            folds[fold][entity]['n_benign'] = np.sum(folds[fold][entity]['y'] == 0)
            folds[fold][entity]['ratio'] = folds[fold][entity]['n_pathogenic'] / folds[fold][entity]['n_benign']

            if entity == 'train':
                # Calculate class weights
                n_samples = folds[fold][entity]['n_SAVs']
                n_classes = 2
                class_weight = {0: n_samples / (n_classes * folds[fold][entity]['n_benign']),
                                1: n_samples / (n_classes * folds[fold][entity]['n_pathogenic'])}
                folds[fold][entity]['class_weight'] = class_weight

    print_format = 'Fold {:d}: \n \
        Train: n_SAVs = {:d}, % SAVs = {:.2%}, n_pathogenic = {:d}, n_benign = {:d}, ratio = {:.2f}:1, n_clusters = {:d}, n_members = {:d}\n \
        Val: n_SAVs = {:d}, % SAVs = {:.2%}, n_pathogenic = {:d}, n_benign = {:d}, ratio = {:.2f}:1, n_clusters = {:d}, n_members = {:d}\n \
        Test: n_SAVs = {:d}, % SAVs = {:.2%}, n_pathogenic = {:d}, n_benign = {:d}, ratio = {:.2f}:1, n_clusters = {:d}, n_members = {:d}'
    
    for fold in range(n_folds):
        msg = print_format.format(fold,
            folds[fold]['train']['n_SAVs'], folds[fold]['train']['percent'], folds[fold]['train']['n_pathogenic'], folds[fold]['train']['n_benign'], folds[fold]['train']['ratio'], folds[fold]['train']['n_clusters'], folds[fold]['train']['n_members'],
            folds[fold]['val']['n_SAVs'],   folds[fold]['val']['percent'],   folds[fold]['val']['n_pathogenic'],   folds[fold]['val']['n_benign'], folds[fold]['val']['ratio'], folds[fold]['val']['n_clusters'],   folds[fold]['val']['n_members'],
            folds[fold]['test']['n_SAVs'],  folds[fold]['test']['percent'],  folds[fold]['test']['n_pathogenic'],  folds[fold]['test']['n_benign'], folds[fold]['test']['ratio'], folds[fold]['test']['n_clusters'],  folds[fold]['test']['n_members'])
        # _LOGGER.error(msg)

    # ************************************************
    # remove_indices = []
    # # Find indices of the target that contains ['Q06187', 'P42680', 'Q08881', 'P07948']
    # for i, t in enumerate(folds[1]['val']['SAV_coords']):
    #     if 'Q06187' in t or 'P42680' in t or 'Q08881' in t or 'P07948' in t:
    #         # Remove t from target_fold_2
    #         remove_indices.append(i)

    # add_indices = []
    # for i, t in enumerate(folds[0]['val']['SAV_coords']):
    #     if 'P12883' in t:
    #         add_indices.append(i)

    # # remove_indices from folds[1]['val']['SAV_coords']
    # folds[1]['val']['SAV_coords'] = [t for i, t in enumerate(folds[1]['val']['SAV_coords']) if i not in remove_indices]

    # # add_indices to folds[1]['val']['SAV_coords']
    # folds[1]['val']['SAV_coords'].extend([folds[0]['val']['SAV_coords'][i] for i in add_indices])
    # _LOGGER.error('No. of SAVs in folds[1][val]: %d' % len(folds[1]['val']['SAV_coords']))
    # _LOGGER.error('Replace Q06187, P42680, Q08881, and P07948 in fold 2 with P12883 in fold 1')
    # # ************************************************

    # for fold in range(n_folds):
    #     msg = print_format.format(fold,
    #         folds[fold]['train']['n_SAVs'], folds[fold]['train']['percent'], folds[fold]['train']['n_pathogenic'], folds[fold]['train']['n_benign'], folds[fold]['train']['ratio'], folds[fold]['train']['n_clusters'], folds[fold]['train']['n_members'],
    #         folds[fold]['val']['n_SAVs'],   folds[fold]['val']['percent'],   folds[fold]['val']['n_pathogenic'],   folds[fold]['val']['n_benign'], folds[fold]['val']['ratio'], folds[fold]['val']['n_clusters'],   folds[fold]['val']['n_members'],
    #         folds[fold]['test']['n_SAVs'],  folds[fold]['test']['percent'],  folds[fold]['test']['n_pathogenic'],  folds[fold]['test']['n_benign'], folds[fold]['test']['ratio'], folds[fold]['test']['n_clusters'],  folds[fold]['test']['n_members'])
    #     _LOGGER.error(msg)

    return folds

def get_rhd_GJB2(feat_path=None, clstr_path=None):
    """Get the features and labels of GJB2 SAVs from Rhapsody dataset
    """
    if feat_path is None or clstr_path is None:
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        feat_path = f'{path}/data/improve/feat_May13.tsv'
        clstr_path = f'{path}/data/improve/c30_clstr_May13.csv'

    df_feat = pd.read_csv(feat_path, sep='\t')
    df_feat['UniProtID'] = df_feat['SAV_coords'].str.split().str[0]
    df_feat = df_feat.drop_duplicates(subset=['SAV_coords'])

    for feat_name in df_feat.columns:
        if df_feat[feat_name].isnull().sum() > 0:
            mean_val = df_feat[feat_name].mean()
            df_feat[feat_name] = df_feat[feat_name].fillna(mean_val)

    n_UniProtID = len(df_feat['UniProtID'].unique())
    n_pathogenic, n_benign = df_feat.labels.value_counts()
    n_SAVs = n_pathogenic + n_benign

    df_clstr = pd.read_csv(clstr_path)
    df_clstr = df_clstr.drop(columns=['rep_member_length', 'member_length', 'member_similarity'], axis=1)
    n_clstr = len(df_clstr)

    _1 = ['P01891', 'P01892', 'P05534', 'P13746', 'P30443', 'P04439']
    _2 = ['P01912', 'P04229', 'P13760', 'P20039', 'P01911']
    _3 = ['P03989', 'P10319', 'P18464', 'P18465', 'P30460', 'P30464', 'P30466', 'P30475', 'P30479', 'P30480', 'P30481', 'P30484', 'P30490', 'P30491', 'P30685', 'Q04826', 'Q31610', 'P01889']
    _4 = ['P04222', 'P30504', 'P30505', 'Q29963', 'Q9TNN7', 'P10321']
    _5 = ['Q6DU44', 'P13747']
    for _no in [_1, _2, _3, _4, _5]:
        _rep_member = df_clstr[df_clstr['member_ID'].str.contains(_no[-1])].rep_member.values[0]
        add_member = ','.join(_no[:-1])
        add_n_member = len(_no[:-1])
        df_clstr.loc[df_clstr['rep_member'] == _rep_member, 'member_ID'] += ',' + add_member
        df_clstr.loc[df_clstr['rep_member'] == _rep_member, 'n_members'] += add_n_member

    data = {}
    for i, row in df_clstr.iterrows():
        data[i] = {}
        clstr = row['member_ID'].split(',')
        df_clstr_feats = df_feat[df_feat['UniProtID'].isin(clstr)]

        data[i]['n_SAVs'] = len(df_clstr_feats)
        data[i]['SAV_coords'] = df_clstr_feats['SAV_coords'].tolist()
        data[i]['n_members'] = row['n_members']
        data[i]['member_ID'] = clstr
        data[i]['y'] = df_clstr_feats['labels'].values

    # Decendingly Sort data by no. SAVs in each cluster
    data_sorted = sorted(data.items(), key=lambda x: x[1]['n_SAVs'], reverse=True)
    data_sorted = [i[1] for i in data_sorted]

    for i in range(len(data_sorted)):
        if 'P29033' in data_sorted[i]['member_ID']:
            P29033 = data_sorted[i]
    return P29033

def main():
    # Get the path of parent of parent directory
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    feat_path = f'{path}/data/improve/feat_May13.tsv'
    clstr_path = f'{path}/data/improve/c30_clstr_May13.csv'
    folds = split_data(feat_path, clstr_path)
    return folds

if __name__ == "__main__":
    main()