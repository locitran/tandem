import os 
import logging
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from ..olds.run import use_all_gpus, get_config, get_seed
from ..olds.run import getR20000, getTestset
from ..utils.settings import FEAT_STATS, dynamics_feat, structure_feat, seq_feat
from ..utils.settings import TANDEM_R20000, TANDEM_GJB2, TANDEM_RYR1, TANDEM_PKD1
from ..utils.settings import ROOT_DIR, CLUSTER
from .modules import Preprocessing, Callback_CSVLogger, DelayedEarlyStopping, BinaryF1Score
from .modules import np_to_dataset , build_optimizer, build_model, plot_acc_loss_3fold_CV
from .config import model_config

LOGGER = logging.getLogger(__name__)

def import_data(TANDEM_testSet):
    ##################### 2. Set up feature set #####################
    df_stats = pd.read_csv(FEAT_STATS)
    t_sel_feats = df_stats.sort_values('ttest_rank').head(33)['feature'].values
    sel_DYNfeats = [feat for feat in t_sel_feats if feat in dynamics_feat.keys()]
    sel_STRfeats = [feat for feat in t_sel_feats if feat in structure_feat.keys()]
    sel_SEQfeats = [feat for feat in t_sel_feats if feat in seq_feat.keys()]
    t_sel_feats = sel_DYNfeats + sel_STRfeats + sel_SEQfeats
    LOGGER.error("*"*50)
    LOGGER.error("Feature selection based on ttest rank")
    LOGGER.error(FEAT_STATS)
    LOGGER.error(f"Feature set: {t_sel_feats}")
    ##################### 3. Set up data #####################
    folds, R20000, preprocess_feat = getR20000(TANDEM_R20000, CLUSTER, feat_names=t_sel_feats)
    test_knw, test_unk = getTestset(TANDEM_testSet, t_sel_feats, preprocess_feat) 
    input_shape = R20000[2].shape[1]
    return folds, R20000, test_knw, test_unk, input_shape

def train_model(TANDEM_testSet,
                name,
                seed=73):
    NAME_OF_EXPERIMENT = f'Direct_train_{name}'
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(ROOT_DIR, 'logs', NAME_OF_EXPERIMENT, f'{name}-{current_time}-seed-{seed}')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=f'{log_dir}/log.txt', level=logging.ERROR, format='%(message)s')
    logging.error("Start Time = %s", current_time)
    use_all_gpus()

    R20000_folds, R20000, test_knw, test_unk, input_shape = import_data(TANDEM_testSet)
    SAV_coords, labels, features = test_knw
    VUS_coords, VUS_labels, VUS_features = test_unk
    labels = np.argmax(labels, axis=1)

    ##################### 3. Set up model configuration #####################
    patience = 50
    n_hidden = 2
    cfg = get_config(input_shape, n_hidden=n_hidden, patience=patience, dropout_rate=0.0,
                     n_neuron_per_hidden=33, n_neuron_last_hidden=33)
    
    cfg['training']['n_epochs'] = 10000
    logging.error("Start from epoch: %d", cfg.training.callbacks.EarlyStopping.start_from_epoch)

    ##################### 5. Split test data #####################
    # 1. Split 3 folds (60% – 30% – 10%)
    train_indices, test_indices = train_test_split(np.arange(len(labels)), test_size=0.1, random_state=seed, stratify=labels)
    kf = StratifiedKFold(n_splits=3, random_state=seed, shuffle=True)
    folds = []
    for i, (train_idx, val_idx) in enumerate(kf.split(train_indices, labels[train_indices])):
        train, val = train_indices[train_idx], train_indices[val_idx]
        test = test_indices
        # Save the folds
        element = {
            'train': {'x': features[train], 'y': labels[train], 'SAV_coords': SAV_coords[train]},
            'val': {'x': features[val], 'y': labels[val], 'SAV_coords': SAV_coords[val]},
            'test': {'x': features[test], 'y': labels[test], 'SAV_coords': SAV_coords[test]}
        }
        folds.append(element)
        # log the folds
        logging.error("Fold %d - Train: %dpos + %dneg, Val: %dpos + %dneg, Test: %dpos + %dneg", i+1, 
                      np.sum(labels[train]), len(train)-np.sum(labels[train]), 
                      np.sum(labels[val]), len(val)-np.sum(labels[val]), 
                      np.sum(labels[test]), len(test)-np.sum(labels[test]))
        logging.error("Train: %s", SAV_coords[train])
        logging.error("Val: %s", SAV_coords[val])
        logging.error("Test: %s", SAV_coords[test])

    evaluations = {}
    for split_idx in range(3):
        get_seed(seed)
        fold = folds[split_idx]
        train, val, test = fold['train'], fold['val'], fold['test']
        x_train, y_train, SAVs_train = train['x'], train['y'], train['SAV_coords']
        x_val, y_val, SAVs_val = val['x'], val['y'], val['SAV_coords']
        x_test, y_test, SAVs_test  = test['x'], test['y'], test['SAV_coords']

        y_train = Preprocessing.one_hot_encoding_labels(y_train, 2)
        y_val = Preprocessing.one_hot_encoding_labels(y_val, 2)
        y_test = Preprocessing.one_hot_encoding_labels(y_test, 2)

        train_ds = np_to_dataset(x_train, y_train, shuffle=True, batch_size=cfg.training.batch_size, seed=seed)
        val_ds = np_to_dataset(x_val, y_val, shuffle=False, batch_size=cfg.training.batch_size, seed=seed)
        test_ds = np_to_dataset(x_test, y_test, shuffle=False, batch_size=cfg.training.batch_size, seed=seed)

        y_knw = Preprocessing.one_hot_encoding_labels(labels, 2)
        knw_ds  = np_to_dataset(features, y_knw, shuffle=False, batch_size=cfg.training.batch_size, seed=seed)
        ##################### 5. Train model on test data #####################
        csv_logger = Callback_CSVLogger(
            data=[train_ds, val_ds], 
            name=['train', 'val'],
            log_file=f'{log_dir}/history_fold_{split_idx}.csv',
        )
        early_stopping = DelayedEarlyStopping(**cfg.training.callbacks.EarlyStopping)

        model = build_model(cfg, verbose=False)
        optimizer = build_optimizer(cfg)
        
        init_weight = f'/mnt/nas_1/YangLab/loci/tandem/models/Direct_train_RYR1/RYR1-20250502-1546-seed-0/model_fold_{split_idx+1}_init.weights.h5'
        model.load_weights(init_weight)
        model.compile(optimizer=optimizer, loss=cfg.training.loss, 
                    metrics=['accuracy', 
                            tf.keras.metrics.AUC(name='auc'), 
                            tf.keras.metrics.Precision(name='precision'), 
                            tf.keras.metrics.Recall(name='recall'),
                            BinaryF1Score(name='f1_score')
                    ])
        # Save model weights
        model.save_weights(f'{log_dir}/model_fold_{split_idx+1}_init.weights.h5')
    
        # Train model on GJB2 data
        model.fit(
            # GJB2_train_ds,
            # validation_data=GJB2_val_ds,
            train_ds,
            validation_data=val_ds,
            epochs=cfg.training.n_epochs,
            callbacks=[csv_logger, early_stopping],
            verbose=1,
            batch_size=cfg.training.batch_size,
        )

        # Evaluation after training
        val_performance = model.evaluate(val_ds)
        test_performance = model.evaluate(test_ds)
        
        evaluations[split_idx] = {
            'val_loss': val_performance[0], 'val_accuracy': val_performance[1], 'val_auc': val_performance[2], 'val_precision': val_performance[3], 'val_recall': val_performance[4], 'val_f1': val_performance[5],
            'test_loss': test_performance[0], 'test_accuracy': test_performance[1], 'test_auc': test_performance[2], 'test_precision': test_performance[3], 'test_recall': test_performance[4], 'test_f1': test_performance[5],
        }
        msg = "Fold %d - val_loss: %.1f, val_accuracy: %.1f%%, val_auc: %.1f, val_precision: %.1f, val_recall: %.1f, val_f1: %.1f, " + \
                "test_loss: %.1f, test_accuracy: %.1f%%, test_auc: %.1f, test_precision: %.1f, test_recall: %.1f, test_f1: %.1f"
        logging.error(msg, split_idx+1, val_performance[0], val_performance[1] * 100, val_performance[2], val_performance[3], val_performance[4], val_performance[5],
                                test_performance[0], test_performance[1] * 100, test_performance[2], test_performance[3], test_performance[4], test_performance[5])
        model.save(f'{log_dir}/model_fold_{split_idx+1}.h5')

    df_evaluations = pd.DataFrame(evaluations).T
    df_evaluations.to_csv(f'{log_dir}/evaluations.csv')

    df_overall = pd.DataFrame(columns=df_evaluations.columns)
    df_overall.loc['mean'] = df_evaluations.mean()
    df_overall.loc['std'] = df_evaluations.std()
    df_overall.loc['sem'] = df_evaluations.sem()
    df_overall.to_csv(f'{log_dir}/overall.csv', index=False)

    logging.error("-----------------------------------------------------------------")
    logging.error("val_loss: %.1f±%.1f, val_accuracy: %.1f±%.1f%%, val_auc: %.1f±%.1f, val_precision: %.1f±%.1f, val_recall: %.1f±%.1f, val_f1: %.1f±%.1f",
                    df_overall.loc['mean', 'val_loss'], df_overall.loc['sem', 'val_loss'], df_overall.loc['mean', 'val_accuracy'] * 100, df_overall.loc['sem', 'val_accuracy'] * 100, df_overall.loc['mean', 'val_auc'], df_overall.loc['sem', 'val_auc'], df_overall.loc['mean', 'val_precision'], df_overall.loc['sem', 'val_precision'], df_overall.loc['mean', 'val_recall'], df_overall.loc['sem', 'val_recall'], df_overall.loc['mean', 'val_f1'], df_overall.loc['sem', 'val_f1'])
    logging.error("test_loss: %.1f±%.1f, test_accuracy: %.1f±%.1f%%, test_auc: %.1f±%.1f, test_precision: %.1f±%.1f, test_recall: %.1f±%.1f, test_f1: %.1f±%.1f",
                    df_overall.loc['mean', 'test_loss'], df_overall.loc['sem', 'test_loss'], df_overall.loc['mean', 'test_accuracy'] * 100, df_overall.loc['sem', 'test_accuracy'] * 100, df_overall.loc['mean', 'test_auc'], df_overall.loc['sem', 'test_auc'], df_overall.loc['mean', 'test_precision'], df_overall.loc['sem', 'test_precision'], df_overall.loc['mean', 'test_recall'], df_overall.loc['sem', 'test_recall'], df_overall.loc['mean', 'val_f1'], df_overall.loc['sem', 'val_f1'])
    logging.error("-----------------------------------------------------------------")

    import matplotlib.pyplot as plt    
    folds_history = [pd.read_csv(f'{log_dir}/history_fold_{j}.csv') for j in range(3)]
    fig = plot_acc_loss_3fold_CV(folds_history, 'Training History')
    fig.savefig(f'{log_dir}/training_history.png')
    plt.close(fig)
