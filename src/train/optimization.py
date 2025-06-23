import os 
import logging
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .run import train_model, use_all_gpus, get_config
from .run import getR20000, getTestset
from ..utils.settings import FEAT_STATS, dynamics_feat, structure_feat, seq_feat
from ..utils.settings import TANDEM_R20000, TANDEM_GJB2, TANDEM_RYR1, CLUSTER
from ..utils.settings import RHAPSODY_R20000, RHAPSODY_GJB2, RHAPSODY_RYR1, CLUSTER
from ..utils.settings import TANDEM_PKD1, CLUSTER, ROOT_DIR, RHAPSODY_FEATS
from .config import model_config

import logging

def simple_training(seed=17):
    NAME_OF_EXPERIMENT = 'Simple_training'
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(ROOT_DIR, 'logs', NAME_OF_EXPERIMENT, current_time)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=f'{log_dir}/log.txt', level=logging.ERROR, format='%(message)s')
    logging.error("Start Time = %s", current_time)
    ##################### 2. Set up feature set #####################
    df_stats = pd.read_csv(FEAT_STATS)
    t_sel_feats = df_stats.sort_values('ttest_rank').head(33)['feature'].values
    sel_DYNfeats = [feat for feat in t_sel_feats if feat in dynamics_feat.keys()]
    sel_STRfeats = [feat for feat in t_sel_feats if feat in structure_feat.keys()]
    sel_SEQfeats = [feat for feat in t_sel_feats if feat in seq_feat.keys()]
    t_sel_feats = sel_DYNfeats + sel_STRfeats + sel_SEQfeats
    logging.error("*"*50)
    logging.error("Feature selection based on ttest rank")
    logging.error(FEAT_STATS)
    logging.error(f"Feature set: {t_sel_feats}")

    ##################### 2. Set up feature files #####################
    logging.error("*"*50)
    logging.error("Feature files")
    logging.error(f"R20000: {TANDEM_R20000}")
    logging.error(f"GJB2: {TANDEM_GJB2}")
    logging.error(f"RYR1: {TANDEM_RYR1}")
    logging.error(f"Cluster path: {CLUSTER}")

    logging.error("Description: TANDEM feature set for R20000, GJB2, and RYR1")
    logging.error("Training DNN with 33 features ranked by ttest")
    logging.error("*"*50)

    ##################### 3. Set up data #####################
    use_all_gpus()
    folds, R20000, preprocess_feat = getR20000(TANDEM_R20000, CLUSTER, feat_names=t_sel_feats)
    GJB2_knw, GJB2_unk = getTestset(TANDEM_GJB2, t_sel_feats, preprocess_feat)
    RYR1_knw, RYR1_unk = getTestset(TANDEM_RYR1, t_sel_feats, preprocess_feat)
    input_shape = R20000[2].shape[1]

    patience = 50
    n_hidden = 5
    cfg = get_config(input_shape, n_hidden=n_hidden, patience=patience, dropout_rate=0.0)

    logging.error("seed = %d" % seed)
    train_model(folds, cfg, log_dir, GJB2_knw, GJB2_unk, RYR1_knw, RYR1_unk, seed=seed)
    logging.error("End Time = %s", datetime.datetime.now().strftime("%Y%m%d-%H%M")) # Write to log
    logging.error("#"*50) # Write to log


def test_numberOflayers_TANDEM(seed=17):
    """We use the ttest ranking method to select 33 features from the feature set."""
    ##################### 1. Set up logging and experiment name #####################
    NAME_OF_EXPERIMENT = 'Optimization_Tandem_NumberOfLayers'

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    n_hidden=6 ; patience=50
    log_dir = os.path.join(ROOT_DIR, 'logs', NAME_OF_EXPERIMENT, current_time)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=f'{log_dir}/log.txt', level=logging.ERROR, format='%(message)s')
    ##################### 2. Set up feature set #####################
    df_stats = pd.read_csv(FEAT_STATS)
    t_sel_feats = df_stats.sort_values('ttest_rank').head(33)['feature'].values
    sel_DYNfeats = [feat for feat in t_sel_feats if feat in dynamics_feat.keys()]
    sel_STRfeats = [feat for feat in t_sel_feats if feat in structure_feat.keys()]
    sel_SEQfeats = [feat for feat in t_sel_feats if feat in seq_feat.keys()]
    t_sel_feats = sel_DYNfeats + sel_STRfeats + sel_SEQfeats
    logging.error("*"*50)
    logging.error("Feature selection based on ttest rank")
    logging.error(FEAT_STATS)
    logging.error(f"Feature set: {t_sel_feats}")

    ##################### 2. Set up feature files #####################
    logging.error("*"*50)
    logging.error("Feature files")
    logging.error(f"R20000: {TANDEM_R20000}")
    logging.error(f"GJB2: {TANDEM_GJB2}")
    logging.error(f"RYR1: {TANDEM_RYR1}")
    logging.error(f"Cluster path: {CLUSTER}")

    logging.error("Description: TANDEM feature set for R20000, GJB2, and RYR1")
    logging.error("Training DNN with 33 features ranked by ttest")
    logging.error("*"*50)

    ##################### 3. Set up data #####################
    use_all_gpus()
    folds, R20000, preprocess_feat = getR20000(TANDEM_R20000, CLUSTER, feat_names=t_sel_feats)
    GJB2_knw, GJB2_unk = getTestset(TANDEM_GJB2, t_sel_feats, preprocess_feat) 
    RYR1_knw, RYR1_unk = getTestset(TANDEM_RYR1, t_sel_feats, preprocess_feat)
    input_shape = R20000[2].shape[1]

    ##################### 3. Set up model configuration #####################
    input_shape = R20000[2].shape[1]
    param_grid = {
        'n_hidden': [5, 6, 8, 10, 12],
    }
    patience = 50
    logging.error("Parameter Grid: \n%s", param_grid) # Write to log

    logging.error("seed = %d" % seed)
    for n_hidden in param_grid['n_hidden']:
        ##################### 4. Set up model configuration #####################
        cfg = get_config(input_shape, n_hidden=n_hidden, patience=patience, dropout_rate=0.0)
        
        ###################### 5. Train model #####################
        train_dir = os.path.join(log_dir, f"n_hidden-{n_hidden}")
        train_model(folds, cfg, train_dir, GJB2_knw, GJB2_unk, RYR1_knw, RYR1_unk, seed=seed)

    logging.error("End Time = %s", datetime.datetime.now().strftime("%Y%m%d-%H%M")) # Write to log
    logging.error("#"*50) # Write to log

def test_numberOflayers_RHAPSODY(seed=17):
    """We use the 33 features from Rhapsody study."""
    ##################### 1. Set up logging and experiment name #####################
    NAME_OF_EXPERIMENT = 'Optimization_Rhapsody_NumberOfLayers'

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    n_hidden=6 ; patience=50
    log_dir = os.path.join(ROOT_DIR, 'logs', NAME_OF_EXPERIMENT, current_time)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=f'{log_dir}/log.txt', level=logging.ERROR, format='%(message)s')
    ##################### 2. Set up feature set #####################
    
    logging.error("*"*50)
    logging.error("Feature selection: all 33 features from Rhapsody")
    logging.error(f"Feature set: {RHAPSODY_FEATS}")

    ##################### 2. Set up feature files #####################
    logging.error("*"*50)
    logging.error("Feature files")
    logging.error(f"R20000: {RHAPSODY_R20000}")
    logging.error(f"GJB2: {RHAPSODY_GJB2}")
    logging.error(f"RYR1: {RHAPSODY_RYR1}")
    logging.error(f"Cluster path: {CLUSTER}")

    logging.error("Description: RHAPSODY feature set for R20000, GJB2, and RYR1")
    logging.error("Training DNN with 33 features ranked by ttest")
    logging.error("*"*50)

    ##################### 3. Set up data #####################
    use_all_gpus()
    folds, R20000, preprocess_feat = getR20000(RHAPSODY_R20000, CLUSTER, feat_names=RHAPSODY_FEATS)
    GJB2_knw, GJB2_unk = getTestset(RHAPSODY_GJB2, RHAPSODY_FEATS, preprocess_feat) 
    RYR1_knw, RYR1_unk = getTestset(RHAPSODY_RYR1, RHAPSODY_FEATS, preprocess_feat)
    input_shape = R20000[2].shape[1]

    ##################### 3. Set up model configuration #####################
    input_shape = R20000[2].shape[1]
    param_grid = {
        'n_hidden': [5, 6, 8, 10, 12],
    }
    patience = 50
    logging.error("Parameter Grid: \n%s", param_grid) # Write to log

    logging.error("seed = %d" % seed)
    for n_hidden in param_grid['n_hidden']:
        ##################### 4. Set up model configuration #####################
        cfg = get_config(input_shape, n_hidden=n_hidden, patience=patience, dropout_rate=0.0)
        
        ###################### 5. Train model #####################
        train_dir = os.path.join(log_dir, f"n_hidden-{n_hidden}")
        train_model(folds, cfg, train_dir, GJB2_knw, GJB2_unk, RYR1_knw, RYR1_unk, seed=seed)

    logging.error("End Time = %s", datetime.datetime.now().strftime("%Y%m%d-%H%M")) # Write to log
    logging.error("#"*50) # Write to log

def visualization_optimization(tandem_dir, rhapsody_dir, hidden_layers=[5, 6, 8, 10, 12]):

    plt.style.use('default')
    imp_history_layers = [os.path.join(tandem_dir, f'n_hidden-{i}') for i in hidden_layers]
    imp_evaluations_layers = [os.path.join(history, 'evaluations.csv') for history in imp_history_layers]
    imp_evaluations_layers = [pd.read_csv(evaluation) for evaluation in imp_evaluations_layers]

    imp_val_accuracy = [evaluation['val_accuracy'] for evaluation in imp_evaluations_layers]
    imp_test_accuracy = [evaluation['test_accuracy'] for evaluation in imp_evaluations_layers]
    imp_GJB2_accuracy = [evaluation['GJB2_notnan_accuracy'] for evaluation in imp_evaluations_layers]
    imp_RYR1_accuracy = [evaluation['RYR1_notnan_accuracy'] for evaluation in imp_evaluations_layers]

    rhd_history_layers = [os.path.join(rhapsody_dir, f'n_hidden-{i}') for i in hidden_layers]
    rhd_evaluations_layers = [os.path.join(history, 'evaluations.csv') for history in rhd_history_layers]
    rhd_evaluations_layers = [pd.read_csv(evaluation) for evaluation in rhd_evaluations_layers]

    rhd_val_accuracy = [evaluation['val_accuracy'] for evaluation in rhd_evaluations_layers]
    rhd_test_accuracy = [evaluation['test_accuracy'] for evaluation in rhd_evaluations_layers]
    rhd_GJB2_accuracy = [evaluation['GJB2_notnan_accuracy'] for evaluation in rhd_evaluations_layers]
    rhd_RYR1_accuracy = [evaluation['RYR1_notnan_accuracy'] for evaluation in rhd_evaluations_layers]

    # T-test comparison between IMPROVE and Rhapsody
    from scipy.stats import ttest_rel
    # Validation
    t_test_val = [ttest_rel(imp_val_accuracy[i], rhd_val_accuracy[i]) for i in range(len(hidden_layers))]
    t_test_test = [ttest_rel(imp_test_accuracy[i], rhd_test_accuracy[i]) for i in range(len(hidden_layers))]
    t_test_GJB2 = [ttest_rel(imp_GJB2_accuracy[i], rhd_GJB2_accuracy[i]) for i in range(len(hidden_layers))]
    t_test_RYR1 = [ttest_rel(imp_RYR1_accuracy[i], rhd_RYR1_accuracy[i]) for i in range(len(hidden_layers))]

    t_test_val = [(t.statistic, t.pvalue) for t in t_test_val]
    t_test_test = [(t.statistic, t.pvalue) for t in t_test_test]
    t_test_GJB2 = [(t.statistic, t.pvalue) for t in t_test_GJB2]
    t_test_RYR1 = [(t.statistic, t.pvalue) for t in t_test_RYR1]
    t_test_pos = [0.825, 0.845, 0.9, 0.8]

    positions = [i for i in range(len(hidden_layers))]
    set_colors = ['r', 'y', 'g', 'b']
    index = np.arange(len(hidden_layers))

    # Plot 4 figure in once: val_accuracy, test_accuracy, GJB2_accuracy, RYR1_accuracy of evaluations_layers 
    # x-axis: hidden_layers, 
    # y-axis 1: val_accuracy, y-axis 2: test_accuracy, y-axis 3: GJB2_accuracy, y-axis 4: RYR1_accuracy
    # 4 rows, 1 column
    fig, axs = plt.subplots(4, 1, figsize=(12, 9))

    # Plot violin plot
    for i, (val, test, GJB2, RYR1) in enumerate(zip(imp_val_accuracy, imp_test_accuracy, imp_GJB2_accuracy, imp_RYR1_accuracy)):
        pos = [i-0.1, i-0.05, i, i+0.05, i+0.1]
        data_set = [val, test, GJB2, RYR1]

        # Plot 5 splits for each data_set
        for row, data in enumerate(data_set):
            for split in range(5):
                axs[row].plot(pos[split], data[split], 'k-o', markersize=4)
                # axs[row].plot(i, data[split], 'k-o', markersize=4)

        # Violin plot for each hidden layer
        for k, data in enumerate(data_set):
            parts = axs[k].violinplot(data, positions=[i], widths=0.3, showmeans=True, showextrema=True)

            # Set thick black border
            parts['bodies'][0].set_facecolor(set_colors[k])
            for pc in parts['bodies']:
                pc.set_linewidth(2)
                pc.set_edgecolor('black')   
                pc.set_linestyle('solid')
                    
            # Set verticle lines color
            for partname in ('cbars', 'cmaxes', 'cmins', 'cmeans'):
                parts[partname].set_edgecolor(set_colors[k])
                parts[partname].set_linewidth(2)
                parts[partname].set_linestyle('solid')
                parts[partname].set_alpha(0.5)

            # Plot the mean value
            axs[k].text(i, np.mean(data), f'{np.mean(data):.3f}', fontsize=12, color='black')

        # Plot t-test result
        for row, t_test in enumerate([t_test_val, t_test_test, t_test_GJB2, t_test_RYR1]):
            if t_test[i][1] < 0.05:
                axs[row].text(i+0.02, t_test_pos[row], f'{t_test[i][1]:.2e}*', fontsize=12, color='black')
            else:
                axs[row].text(i+0.02, t_test_pos[row], f'{t_test[i][1]:.2e}', fontsize=12, color='black')
            axs[row].text(-0.35, t_test_pos[row], 'p-value', fontsize=12, color='black')

    # Plot violin plot
    distance = 0.3
    for i, (val, test, GJB2, RYR1) in enumerate(zip(rhd_val_accuracy, rhd_test_accuracy, rhd_GJB2_accuracy, rhd_RYR1_accuracy)):
        pos = [i-0.1, i-0.05, i, i+0.05, i+0.1]
        data_set = [val, test, GJB2, RYR1]

        # Plot 5 splits for each data_set
        for row, (data, color) in enumerate(zip(data_set, set_colors)):
            # Plot data
            for split in range(5):
                axs[row].plot(pos[split]+distance, data[split], 'k-x', markersize=4)
            
        # Violin plot for each hidden layer
        for k, data in enumerate(data_set):
            parts = axs[k].violinplot(data, positions=[i+distance], widths=0.3, showmeans=True, showextrema=True)

            # Set color 
            parts['bodies'][0].set_facecolor(set_colors[k])
            # Set thick black border
            for pc in parts['bodies']:
                pc.set_linewidth(2)
                pc.set_edgecolor('black')   
                pc.set_linestyle('solid')
                
                # Set violine hatch
                pc.set_hatch('/')
                    
            # Set verticle lines color
            for partname in ('cbars', 'cmaxes', 'cmins', 'cmeans'):
                parts[partname].set_edgecolor(set_colors[k])
                parts[partname].set_linewidth(2)
                parts[partname].set_linestyle('solid')
                parts[partname].set_alpha(0.5)

            # Plot the mean value
            axs[k].text(i+distance, np.mean(data), f'{np.mean(data):.3f}', fontsize=12, color='black')
            
    for i, ax in enumerate(axs):
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_xticks(ticks=index) if ax == axs[-1] else ax.xaxis.set_visible(False)
        ax.set_xticklabels(hidden_layers) if ax == axs[-1] else ax.set_xticklabels([])
        ax.set_ylabel([r'R20000$_{val}$', r'R20000$_{test}$', r'GJB2$_{knw}$', r'RYR1$_{kwn}$'][i], fontsize=15)

    # y-axis limit row 0, 1, 2, 3
    axs[0].set_ylim(bottom=None, top=0.84)
    axs[1].set_ylim(bottom=None, top=0.85)
    axs[2].set_ylim(bottom=None, top=0.95)
    axs[3].set_ylim(bottom=0.63, top=0.82)

    # # plot nothing, just add labels
    axs[-1].scatter([], [], color='k', marker='o', label='TANDEM')
    axs[-1].scatter([], [], color='k', marker='x', label=r'Rhapsody$_{DNN}$')
    axs[-1].bar(2, 0, color='w', edgecolor='black', hatch='', label='TANDEM')
    axs[-1].bar(2, 0, color='w', edgecolor='black', hatch='///', label=r'Rhapsody$_{DNN}$')

    # fig.suptitle('Model performance with different hidden layers', fontsize=20)
    fig.supxlabel('No. hidden layers', fontsize=15)
    fig.supylabel('Accuracy', fontsize=15)

    # fig.legend(fontsize=10, loc=[0.898, 0.8], ncol=1)
    fig.legend(fontsize=12, loc=[0.27, 0.095], ncol=4)
    plt.xticks(ticks=index, labels=hidden_layers, fontsize=15)
    fig.tight_layout()
    # plt.show()

    # SAve
    folder = '.'
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, 'optimization.png'), dpi=300, bbox_inches='tight')


def test_ranking_method(seed=17):
    """We use the ttest ranking method to select 33 features from the feature set."""
    ##################### 1. Set up logging and experiment name #####################
    NAME_OF_EXPERIMENT = 'Optimization_Tandem_RankingMethod'

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    n_hidden=5 ; patience=50
    log_dir = os.path.join(ROOT_DIR, 'logs', NAME_OF_EXPERIMENT, current_time)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=f'{log_dir}/log.txt', level=logging.ERROR, format='%(message)s')
    ##################### 2. Set up feature files #####################
    logging.error("*"*50)
    logging.error("Feature files")
    logging.error(f"R20000: {TANDEM_R20000}")
    logging.error(f"GJB2: {TANDEM_GJB2}")
    logging.error(f"RYR1: {TANDEM_RYR1}")
    logging.error(f"Cluster path: {CLUSTER}")
    logging.error("Description: TANDEM feature set for R20000, GJB2, and RYR1")
    logging.error("Training DNN with 33 features ranked by ttest")
    logging.error(f"seed = {seed}")
    logging.error(f"Number of hidden layers = {n_hidden}")
    logging.error("*"*50)
    use_all_gpus()
    
    ### T-test
    ##################### 2. Set up feature set #####################
    df_stats = pd.read_csv(FEAT_STATS)
    t_sel_feats = df_stats.sort_values('ttest_rank').head(33)['feature'].values
    sel_DYNfeats = [feat for feat in t_sel_feats if feat in dynamics_feat.keys()]
    sel_STRfeats = [feat for feat in t_sel_feats if feat in structure_feat.keys()]
    sel_SEQfeats = [feat for feat in t_sel_feats if feat in seq_feat.keys()]
    t_sel_feats = sel_DYNfeats + sel_STRfeats + sel_SEQfeats
    logging.error("*"*50)
    logging.error("# Feature selection based on ttest rank")
    logging.error(FEAT_STATS)
    logging.error(f"Feature set: {t_sel_feats} {len(t_sel_feats)}")
    ##################### 3. Set up data #####################
    folds, R20000, preprocess_feat = getR20000(TANDEM_R20000, CLUSTER, feat_names=t_sel_feats, folder=log_dir)
    GJB2_knw, GJB2_unk = getTestset(TANDEM_GJB2, t_sel_feats, preprocess_feat) 
    RYR1_knw, RYR1_unk = getTestset(TANDEM_RYR1, t_sel_feats, preprocess_feat)
    input_shape = R20000[2].shape[1]
    ##################### 4. Set up model configuration #####################
    cfg = get_config(input_shape, n_hidden=n_hidden, patience=patience, dropout_rate=0.0)
    ###################### 5. Train model #####################
    train_dir = os.path.join(log_dir, 'ttest')
    os.makedirs(train_dir, exist_ok=True)
    train_model(folds, cfg, train_dir, GJB2_knw, GJB2_unk, RYR1_knw, RYR1_unk, seed=seed)

    ### T-test + KS-test
    ##################### 2. Set up feature set #####################
    df_stats = pd.read_csv(FEAT_STATS)
    t_sel_feats = df_stats.sort_values('sum_rank_ttestANDKStest').head(33)['feature'].values
    sel_DYNfeats = [feat for feat in t_sel_feats if feat in dynamics_feat.keys()]
    sel_STRfeats = [feat for feat in t_sel_feats if feat in structure_feat.keys()]
    sel_SEQfeats = [feat for feat in t_sel_feats if feat in seq_feat.keys()]
    t_sel_feats = sel_DYNfeats + sel_STRfeats + sel_SEQfeats
    logging.error("*"*50)
    logging.error("# Feature selection based on sum_rank_ttestANDKStest")
    logging.error(FEAT_STATS)
    logging.error(f"Feature set: {t_sel_feats} {len(t_sel_feats)}")
    ##################### 3. Set up data #####################
    folds, R20000, preprocess_feat = getR20000(TANDEM_R20000, CLUSTER, feat_names=t_sel_feats, folder=log_dir)
    GJB2_knw, GJB2_unk = getTestset(TANDEM_GJB2, t_sel_feats, preprocess_feat) 
    RYR1_knw, RYR1_unk = getTestset(TANDEM_RYR1, t_sel_feats, preprocess_feat)
    input_shape = R20000[2].shape[1]
    ##################### 4. Set up model configuration #####################
    cfg = get_config(input_shape, n_hidden=n_hidden, patience=patience, dropout_rate=0.0)
    ###################### 5. Train model #####################
    train_dir = os.path.join(log_dir, 'ttest+ks-test')
    os.makedirs(train_dir, exist_ok=True)
    train_model(folds, cfg, train_dir, GJB2_knw, GJB2_unk, RYR1_knw, RYR1_unk, seed=seed)

    ### Wasserstein distance
    ##################### 2. Set up feature set #####################
    df_stats = pd.read_csv(FEAT_STATS)
    t_sel_feats = df_stats.sort_values('wasserstein_dist_rank').head(33)['feature'].values
    sel_DYNfeats = [feat for feat in t_sel_feats if feat in dynamics_feat.keys()]
    sel_STRfeats = [feat for feat in t_sel_feats if feat in structure_feat.keys()]
    sel_SEQfeats = [feat for feat in t_sel_feats if feat in seq_feat.keys()]
    t_sel_feats = sel_DYNfeats + sel_STRfeats + sel_SEQfeats
    logging.error("*"*50)
    logging.error("# Feature selection based on wasserstein_dist_rank")
    logging.error(FEAT_STATS)
    logging.error(f"Feature set: {t_sel_feats}")
    ##################### 3. Set up data #####################
    folds, R20000, preprocess_feat = getR20000(TANDEM_R20000, CLUSTER, feat_names=t_sel_feats, folder=log_dir)
    GJB2_knw, GJB2_unk = getTestset(TANDEM_GJB2, t_sel_feats, preprocess_feat) 
    RYR1_knw, RYR1_unk = getTestset(TANDEM_RYR1, t_sel_feats, preprocess_feat)
    input_shape = R20000[2].shape[1]
    ##################### 4. Set up model configuration #####################
    cfg = get_config(input_shape, n_hidden=n_hidden, patience=patience, dropout_rate=0.0)
    ###################### 5. Train model #####################
    train_dir = os.path.join(log_dir, 'wasserstein')
    os.makedirs(train_dir, exist_ok=True)
    train_model(folds, cfg, train_dir, GJB2_knw, GJB2_unk, RYR1_knw, RYR1_unk, seed=seed)

    fns = [
        "consurf", "wtPSIC", "deltaPSIC", "entropy", "ACNR", "SASA", "BLOSUM", "ANM_stiffness_chain",
        "loop_percent", "AG1", "GNM_V2_full", "GNM_co_rank_full", "AG3", "AG5", "Dcom", "GNM_V1_full",
        "GNM_rankV2_full", "GNM_Eigval1_full", "ranked_MI", "DELTA_Hbond", "phobic_percent", "GNM_Eigval2_full",
        "sheet_percent", "Rg", "deltaPolarity", "Lside", "helix_percent", "deltaLside", "ANM_effectiveness_chain",
        "GNM_rankV1_full", "GNM_rmsf_overall_full", "deltaCharge", "delta_phobic_percent"]

    ### thesis feature set
    ##################### 2. Set up feature set #####################
    logging.error("*"*50)
    logging.error("# Feature selection based on thesis selections")
    logging.error(f"Feature set: {fns}")
    ##################### 3. Set up data #####################
    folds, R20000, preprocess_feat = getR20000(TANDEM_R20000, CLUSTER, feat_names=fns, folder=log_dir)
    GJB2_knw, GJB2_unk = getTestset(TANDEM_GJB2, fns, preprocess_feat) 
    RYR1_knw, RYR1_unk = getTestset(TANDEM_RYR1, fns, preprocess_feat)
    input_shape = R20000[2].shape[1]
    ##################### 4. Set up model configuration #####################
    cfg = get_config(input_shape, n_hidden=n_hidden, patience=patience, dropout_rate=0.0)
    ###################### 5. Train model #####################
    train_dir = os.path.join(log_dir, 'thesis_select')
    os.makedirs(train_dir, exist_ok=True)
    train_model(folds, cfg, train_dir, GJB2_knw, GJB2_unk, RYR1_knw, RYR1_unk, seed=seed)

def test_batch_size(seed=17):
    """We use the ttest ranking method to select 33 features from the feature set."""
    ##################### 1. Set up logging and experiment name #####################
    NAME_OF_EXPERIMENT = 'Optimization_Tandem_BatchSize'

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    n_hidden=5 ; patience=50
    log_dir = os.path.join(ROOT_DIR, 'logs', NAME_OF_EXPERIMENT, current_time)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=f'{log_dir}/log.txt', level=logging.ERROR, format='%(message)s')
    ##################### 2. Set up feature files #####################
    logging.error("*"*50)
    logging.error("Feature files")
    logging.error(f"R20000: {TANDEM_R20000}")
    logging.error(f"GJB2: {TANDEM_GJB2}")
    logging.error(f"RYR1: {TANDEM_RYR1}")
    logging.error(f"Cluster path: {CLUSTER}")
    logging.error("Description: TANDEM feature set for R20000, GJB2, and RYR1")
    logging.error("Training DNN with 33 features ranked by ttest")
    logging.error(f"seed = {seed}")
    logging.error(f"Number of hidden layers = {n_hidden}")
    logging.error("*"*50)
    use_all_gpus()
    
    ### T-test
    ##################### 2. Set up feature set #####################
    df_stats = pd.read_csv(FEAT_STATS)
    t_sel_feats = df_stats.sort_values('ttest_rank').head(33)['feature'].values
    sel_DYNfeats = [feat for feat in t_sel_feats if feat in dynamics_feat.keys()]
    sel_STRfeats = [feat for feat in t_sel_feats if feat in structure_feat.keys()]
    sel_SEQfeats = [feat for feat in t_sel_feats if feat in seq_feat.keys()]
    t_sel_feats = sel_DYNfeats + sel_STRfeats + sel_SEQfeats
    logging.error("*"*50)
    logging.error("# Feature selection based on ttest rank")
    logging.error(FEAT_STATS)
    logging.error(f"Feature set: {t_sel_feats} {len(t_sel_feats)}")
    ##################### 3. Set up data #####################
    folds, R20000, preprocess_feat = getR20000(TANDEM_R20000, CLUSTER, feat_names=t_sel_feats, folder=log_dir)
    GJB2_knw, GJB2_unk = getTestset(TANDEM_GJB2, t_sel_feats, preprocess_feat) 
    RYR1_knw, RYR1_unk = getTestset(TANDEM_RYR1, t_sel_feats, preprocess_feat)
    input_shape = R20000[2].shape[1]
    ##################### 4. Set up model configuration #####################
    cfg = get_config(input_shape, n_hidden=n_hidden, patience=patience, dropout_rate=0.0)
    ###################### 5. Train model #####################
    batch_size = [64, 128, 256, 300, 512]
    for bs in batch_size:
        # if bs != 300:
            # continue
        cfg['training']['batch_size'] = bs
        logging.error(f"Batch size = {bs}")
        train_dir = os.path.join(log_dir, f'batch_size_{bs}')
        os.makedirs(train_dir, exist_ok=True)
        train_model(folds, cfg, train_dir, GJB2_knw, GJB2_unk, RYR1_knw, RYR1_unk, seed=seed)

def test_different_numberOfneurons(seed=17):
    ##################### 1. Set up logging and experiment name #####################
    NAME_OF_EXPERIMENT = 'Optimization_Tandem_numberOfneurons'
    patience = 50
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(ROOT_DIR, 'logs', NAME_OF_EXPERIMENT, current_time)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=f'{log_dir}/log.txt', level=logging.ERROR, format='%(message)s')
    ##################### 2. Set up feature files #####################
    logging.error("*"*50)
    logging.error("Feature files")
    logging.error(f"R20000: {TANDEM_R20000}")
    logging.error(f"GJB2: {TANDEM_GJB2}")
    logging.error(f"RYR1: {TANDEM_RYR1}")
    logging.error(f"Cluster path: {CLUSTER}")
    logging.error("Description: TANDEM feature set for R20000, GJB2, and RYR1")
    logging.error("Training DNN with different number of neurons")
    logging.error(f"seed = {seed}")
    logging.error("*"*50)
    use_all_gpus()
    ##################### 3. Set up data #####################
    ### T-test
    ##################### 2. Set up feature set #####################
    df_stats = pd.read_csv(FEAT_STATS)
    t_sel_feats = df_stats.sort_values('ttest_rank').head(33)['feature'].values
    sel_DYNfeats = [feat for feat in t_sel_feats if feat in dynamics_feat.keys()]
    sel_STRfeats = [feat for feat in t_sel_feats if feat in structure_feat.keys()]
    sel_SEQfeats = [feat for feat in t_sel_feats if feat in seq_feat.keys()]
    t_sel_feats = sel_DYNfeats + sel_STRfeats + sel_SEQfeats
    logging.error("*"*50)
    logging.error("# Feature selection based on ttest rank")
    logging.error(FEAT_STATS)
    logging.error(f"Feature set: {t_sel_feats} {len(t_sel_feats)}")
    ##################### 3. Set up data #####################
    folds, R20000, preprocess_feat = getR20000(TANDEM_R20000, CLUSTER, feat_names=t_sel_feats, folder=log_dir)
    GJB2_knw, GJB2_unk = getTestset(TANDEM_GJB2, t_sel_feats, preprocess_feat) 
    RYR1_knw, RYR1_unk = getTestset(TANDEM_RYR1, t_sel_feats, preprocess_feat)
    input_shape = R20000[2].shape[1]
    ##################### 4. Set up model configuration #####################
    array = [
        [33, 33, 33, 33, 10],
        [64, 32, 16, 8, 4],
        [64, 32, 32, 16, 8, 4],
        [64, 64, 32, 16, 8, 4],
        [64, 128, 64, 32, 16, 8]
    ]
    dropout_rate = 0.0
    input_shape = 33
    cfg = model_config()
    cfg.model.input.n_neurons = input_shape
    cfg['model']['input']['dropout_rate'] = dropout_rate

    # No. of neurons in the output layer
    for _list in array:
        for item in cfg.model.hidden:
            del cfg['model']['hidden'][item]
        for i, n_neuron in enumerate(_list):
            hidden_name = f'hidden_{i:02d}'
            cfg['model']['hidden'][hidden_name] = {
                'activation': 'gelu',
                'batch_norm': False,
                'dropout_rate': dropout_rate,
                'initializer': 'glorot_uniform',
                'l1': 0,
                'l2': 0.0001,
                'n_neurons': n_neuron
            }
        cfg['training']['callbacks']['EarlyStopping']['patience'] = patience

        from prettytable import PrettyTable
        # print cfg as table
        tb = PrettyTable()
        tb.field_names = ["Layer", "Activation", "Batch Norm", "Dropout Rate", "Initializer", "L1", "L2", "N Neurons"]
        tb.add_row(["Input", "-", "-", cfg.model.input.dropout_rate, "-", "-", "-", cfg.model.input.n_neurons])
        for layer in cfg.model.hidden:
            tb.add_row([layer, cfg.model.hidden[layer].activation, cfg.model.hidden[layer].batch_norm, cfg.model.hidden[layer].dropout_rate, cfg.model.hidden[layer].initializer, cfg.model.hidden[layer].l1, cfg.model.hidden[layer].l2, cfg.model.hidden[layer].n_neurons])
        tb.add_row(["Output", cfg.model.output.activation, "-", "-", "-", "-", "-", cfg.model.output.n_neurons])
        logging.error("Model Configuration: \n%s", tb) # Write to log

        # print training and optimizer
        tb = PrettyTable()
        tb.field_names = ["Training", "Batch Size", "N Epochs", "Loss", "Metrics"]
        tb.add_row(["Training", cfg.training.batch_size, cfg.training.n_epochs, cfg.training.loss, cfg.training.metrics])
        tb.add_row(["Optimizer", cfg.optimizer.learning_rate, cfg.optimizer.name, "-", "-"])
        logging.error("Training Configuration: \n%s", tb) # Write to log

        ###################### 5. Train model #####################
        _list = '_'.join([str(i) for i in _list])
        logging.error(f"Number of neurons: {_list}")
        train_dir = os.path.join(log_dir, f'neurons_{_list}')
        os.makedirs(train_dir, exist_ok=True)
        train_model(folds, cfg, train_dir, GJB2_knw, GJB2_unk, RYR1_knw, RYR1_unk, seed=seed)
    logging.error("End Time = %s", datetime.datetime.now().strftime("%Y%m%d-%H%M")) # Write to log
    logging.error("#"*50) # Write to log
    # test_different_numberOfneurons()