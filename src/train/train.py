import os
import datetime
import logging
import pandas as pd
import numpy as np
import random
from .modules import Preprocessing, DelayedEarlyStopping, Callback_CSVLogger, BinaryF1Score
from .modules import build_model, np_to_dataset, build_optimizer, plot_acc_loss, build_model_from_config, plot_acc_loss_3fold_CV
from .split_data import split_data
from .config import model_config
import tensorflow as tf

from .run import train_model, use_all_gpus, get_config
from .run import getR20000, getTestset
from ..utils.settings import FEAT_STATS, dynamics_feat, structure_feat, seq_feat
from ..utils.settings import TANDEM_R20000, TANDEM_GJB2, TANDEM_RYR1, CLUSTER
from ..utils.settings import RHAPSODY_R20000, RHAPSODY_GJB2, RHAPSODY_RYR1, CLUSTER
from ..utils.settings import TANDEM_PKD1, CLUSTER, ROOT_DIR, RHAPSODY_FEATS
from .config import model_config
from ..features import TANDEM_FEATS
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

filedir = os.path.dirname(os.path.abspath(__file__))

def is_model_compiled(model):
    return hasattr(model, 'optimizer') and model.optimizer is not None

def get_seed(seed=150):
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    return seed

def train_model(train_ds, val_ds, cfg, folder, filename, model_input=None, seed=None, initial_biase=None):
    """
    Trains a deep learning model on the given training and validation datasets, with optional transfer learning.

    Parameters:
    -----------
    folder : str
        Directory where the model and training history CSV will be saved.

    filename : str
        Base name (without extension) for saving the model file and CSV log.

    model_input : keras.Model or None (default=None)
        If None, the model is trained from scratch using the configuration.
        If a model is provided, transfer learning will be performed on top of the existing model.

    Returns:
    --------
    model : keras.Model
        The trained Keras model after applying early stopping.

    Notes:
    ------
    - Uses Nadam optimizer and categorical cross-entropy loss.
    - Logs training and validation metrics (accuracy, AUC, precision, recall, F1-score) per epoch.
    - Saves the model to '{folder}/{filename}.h5'.
    - Saves the training log to '{folder}/history_{filename}.csv'.
    """
    # Set seed for each fold => Every fold has the same weight initialization
    # Reproducibility
    seed = get_seed(seed)
    
    # Build model
    if model_input is not None:
        model = model_input
    else:
        model = build_model(cfg, initial_biase)
    optimizer = tf.keras.optimizers.Nadam(learning_rate=cfg.optimizer.learning_rate)
    optimizer.build(model.trainable_variables)
    
    csv_logger = Callback_CSVLogger(
        data=[train_ds, val_ds],
        name=['train', 'val'],
        log_file=f'{folder}/history_{filename}.csv'
    )
    early_stopping = DelayedEarlyStopping(**cfg.training.callbacks.EarlyStopping)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 
            tf.keras.metrics.AUC(name='auc'), 
            tf.keras.metrics.Precision(name='precision'), 
            tf.keras.metrics.Recall(name='recall'),
            BinaryF1Score(name='f1_score')
        ]
    )
    model.fit(
        train_ds,
        epochs=cfg.training.n_epochs,
        validation_data=val_ds,
        callbacks=[early_stopping, csv_logger],
        batch_size=300,
    )
    model.save(os.path.join(folder, f'{filename}.h5'), include_optimizer=True)
    return model
    
    
def reproduce_foundation_model(name='reproduce_foundation_model'):
    """
    folds: dictionary contains information of 5 folds
    R20000: all SAVs (SAV_coords, features, labels) that are distributed into folds
    preprocess_feat: class object which helps to standardize new input (e.g., RYR1, GJB2)
    """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(ROOT_DIR, f'logs/{name}/{current_time}')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=f'{log_dir}/log.txt', level=logging.ERROR, format='%(message)s')
    logging.error("Start Time = %s", current_time)
    logging.error("Tensorflow Version: %s", tf.__version__) # Write to log    
    logging.error("Tensorflow Version should be 2.17")
    ##################### 1. Set up feature set #####################
    t_sel_feats = TANDEM_FEATS['v1.1']
    logging.error(f"Feature set: {t_sel_feats}")
    ##################### 2. Set up configuration and data set ######
    seed=17
    patience = 50
    n_hidden = 5
    
    use_all_gpus()
    folds, R20000, preprocess_feat = getR20000(TANDEM_R20000, CLUSTER, feat_names=t_sel_feats)
    GJB2_knw, GJB2_unk = getTestset(TANDEM_GJB2, t_sel_feats, preprocess_feat)
    RYR1_knw, RYR1_unk = getTestset(TANDEM_RYR1, t_sel_feats, preprocess_feat)
    GJB2_notnan_SAV_coords, GJB2_notnan_labels, GJB2_notnan_features = GJB2_knw
    RYR1_notnan_SAV_coords, RYR1_notnan_labels, RYR1_notnan_features = RYR1_knw
    GJB2_nan_SAV_coords, GJB2_nan_labels, GJB2_nan_features = GJB2_unk
    RYR1_nan_SAV_coords, RYR1_nan_labels, RYR1_nan_features = RYR1_unk
    
    # Convert numpy array to tensorflow dataset
    GJB2_nan_ds = np_to_dataset(GJB2_nan_features, GJB2_nan_labels, shuffle=False, batch_size=300, seed=seed)
    GJB2_notnan_ds = np_to_dataset(GJB2_notnan_features, GJB2_notnan_labels, shuffle=False, batch_size=300, seed=seed)
    RYR1_nan_ds = np_to_dataset(RYR1_nan_features, RYR1_nan_labels, shuffle=False, batch_size=300, seed=seed)
    RYR1_notnan_ds = np_to_dataset(RYR1_notnan_features, RYR1_notnan_labels, shuffle=False, batch_size=300, seed=seed)
    
    input_shape = R20000[2].shape[1]
    cfg = get_config(input_shape, n_hidden=n_hidden, patience=patience, dropout_rate=0.0)

    ##################### 3. Train ####################################
    evaluations = {}
    for i, fold in folds.items():
        train, val, test = fold['train'], fold['val'], fold['test']
        train_ds = np_to_dataset(train['x'], train['y'], shuffle=True, batch_size=300, seed=seed)
        val_ds = np_to_dataset(val['x'], val['y'], shuffle=False, batch_size=300, seed=seed)
        test_ds = np_to_dataset(test['x'], test['y'], shuffle=False, batch_size=300, seed=seed)
        initial_biase = np.log([np.sum(train['y'][:, 0]) / np.sum(train['y'][:, 1])]) # Ref: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        model = train_model(
            train_ds, val_ds, cfg=cfg,
            folder=log_dir, filename=f'fold_{i+1}', model_input=None, 
            seed=seed, initial_biase=initial_biase
        )
        
        ### Evaluation
        val_rs = model.evaluate(val_ds)
        test_rs = model.evaluate(test_ds)
        GJB2_notnan_rs = model.evaluate(GJB2_notnan_ds)
        RYR1_notnan_rs = model.evaluate(RYR1_notnan_ds)

        msg = "Fold %d - val_loss: %.2f, val_accuracy: %.1f%%, val_auc: %.2f, val_precision: %.2f, val_recall: %.2f, val_f1: %.2f, " + \
            "test_loss: %.2f, test_accuracy: %.1f%%, test_auc: %.2f, test_precision: %.2f, test_recall: %.2f, test_f1: %.2f, " + \
            "RYR1_loss: %.2f, RYR1_accuracy: %.1f%%, RYR1_auc: %.2f, RYR1_precision: %.2f, RYR1_recall: %.2f, GJB2_notnan_f1: %.2f, " + \
            "GJB2_loss: %.2f, GJB2_accuracy: %.1f%%, GJB2_auc: %.2f, GJB2_precision: %.2f, GJB2_recall: %.2f, RYR1_notnan_f1: %.2f"
        logging.error(
            msg, i+1, 
            val_rs[0], val_rs[1] * 100, val_rs[2], val_rs[3], val_rs[4], val_rs[5],
            test_rs[0], test_rs[1] * 100, test_rs[2], test_rs[3], test_rs[4], test_rs[5], 
            RYR1_notnan_rs[0], RYR1_notnan_rs[1] * 100, RYR1_notnan_rs[2], RYR1_notnan_rs[3], RYR1_notnan_rs[4], GJB2_notnan_rs[5],
            GJB2_notnan_rs[0], GJB2_notnan_rs[1] * 100, GJB2_notnan_rs[2], GJB2_notnan_rs[3], GJB2_notnan_rs[4], RYR1_notnan_rs[5]
        )
        evaluations[i] = {
            'val_loss': val_rs[0], 'val_accuracy': val_rs[1], 'val_auc': val_rs[2], 'val_precision': val_rs[3], 'val_recall': val_rs[4], 'val_f1': val_rs[5],
            'test_loss': test_rs[0], 'test_accuracy': test_rs[1], 'test_auc': test_rs[2], 'test_precision': test_rs[3], 'test_recall': test_rs[4], 'test_f1': test_rs[5],
            'GJB2_notnan_loss': GJB2_notnan_rs[0], 'GJB2_notnan_accuracy': GJB2_notnan_rs[1], 'GJB2_notnan_auc': GJB2_notnan_rs[2], 'GJB2_notnan_precision': GJB2_notnan_rs[3], 'GJB2_notnan_recall': GJB2_notnan_rs[4], 'GJB2_notnan_f1': GJB2_notnan_rs[5],
            'RYR1_notnan_loss': RYR1_notnan_rs[0], 'RYR1_notnan_accuracy': RYR1_notnan_rs[1], 'RYR1_notnan_auc': RYR1_notnan_rs[2], 'RYR1_notnan_precision': RYR1_notnan_rs[3], 'RYR1_notnan_recall': RYR1_notnan_rs[4], 'RYR1_notnan_f1': RYR1_notnan_rs[5],
        }
    
    df_evaluations = pd.DataFrame(evaluations).T
    df_evaluations.to_csv(f'{log_dir}/evaluations.csv', index=False)

    # df_overall: mean row, std row, sem row
    df_overall = pd.DataFrame(columns=df_evaluations.columns)
    df_overall.loc['mean'] = df_evaluations.mean()
    df_overall.loc['std'] = df_evaluations.std()
    df_overall.loc['sem'] = df_evaluations.sem()
    df_overall.to_csv(f'{log_dir}/overall.csv', index=False)

    logging.error("-----------------------------------------------------------------")
    logging.error("Vali - loss: %.2f±%.2f, accuracy: %.1f±%.1f%%, auc: %.2f±%.2f", df_overall.loc['mean', 'val_loss'], df_overall.loc['sem', 'val_loss'], df_overall.loc['mean', 'val_accuracy'] * 100, df_overall.loc['sem', 'val_accuracy'] * 100, df_overall.loc['mean', 'val_auc'], df_overall.loc['sem', 'val_auc'])
    logging.error("Test - loss: %.2f±%.2f, accuracy: %.1f±%.1f%%, auc: %.2f±%.2f", df_overall.loc['mean', 'test_loss'], df_overall.loc['sem', 'test_loss'], df_overall.loc['mean', 'test_accuracy'] * 100, df_overall.loc['sem', 'test_accuracy'] * 100, df_overall.loc['mean', 'test_auc'], df_overall.loc['sem', 'test_auc'])
    logging.error("GJB2 - loss: %.2f±%.2f, accuracy: %.1f±%.1f%%, auc: %.2f±%.2f", df_overall.loc['mean', 'GJB2_notnan_loss'], df_overall.loc['sem', 'GJB2_notnan_loss'], df_overall.loc['mean', 'GJB2_notnan_accuracy'] * 100, df_overall.loc['sem', 'GJB2_notnan_accuracy'] * 100, df_overall.loc['mean', 'GJB2_notnan_auc'], df_overall.loc['sem', 'GJB2_notnan_auc'])
    logging.error("RYR1 - loss: %.2f±%.2f, accuracy: %.1f±%.1f%%, auc: %.2f±%.2f", df_overall.loc['mean', 'RYR1_notnan_loss'], df_overall.loc['sem', 'RYR1_notnan_loss'], df_overall.loc['mean', 'RYR1_notnan_accuracy'] * 100, df_overall.loc['sem', 'RYR1_notnan_accuracy'] * 100, df_overall.loc['mean', 'RYR1_notnan_auc'], df_overall.loc['sem', 'RYR1_notnan_auc'])
    logging.error("-----------------------------------------------------------------")
    # plot
    folds_history = [pd.read_csv(f'{log_dir}/history_fold_{i}.csv') for i in range(1, 6)]
    fig = plot_acc_loss(folds_history, 'Training History')
    fig.savefig(f'{log_dir}/training_history.png')
    fig.close()
    
def reproduce_transfer_learning_model(
    features,
    labels,
    name='reproduce_transfer_learning_model', 
    model_input=None, 
    seed=73,
    patience = 50,
    n_epochs=300,
    start_from_epoch = 10
):
    # Defalut model input : TANDEM_1
    if model_input is None:
        model_input = os.path.join(
            filedir, '..', '..',
            'models/different_number_of_layers/20250423-1234-tandem/n_hidden-5/model_fold_1.h5'
        )
        model_input = os.path.normpath(model_input)
    ##################### 1. Set up logging #########################
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(ROOT_DIR, f'logs/{name}/{current_time}')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=f'{log_dir}/log.txt', level=logging.ERROR, format='%(message)s')
    logging.error("Start Time = %s", current_time)
    logging.error("Tensorflow Version: %s", tf.__version__) # Write to log    
    logging.error("Tensorflow Version should be 2.17")
    ##################### 1. Set up feature set #####################
    t_sel_feats = TANDEM_FEATS['v1.1']
    logging.error(f"Feature set: {t_sel_feats}")
    ##################### 2. Set up data set ######
    _, _, preprocess_feat = getR20000(TANDEM_R20000, CLUSTER, feat_names=t_sel_feats)
    # Fill missing data by mean value and standarization
    features = preprocess_feat.fill_na_mean(features)
    features = preprocess_feat.normalize(features)
    
    n_pathogenic = np.sum(labels)
    n_benign = labels.shape[0] - n_pathogenic
    logging.error("No. %d SAVs %d (benign), %d (pathogenic)", features.shape[0], n_benign, n_pathogenic)

    ##################### 3. Set up model configuration #####################
    cfg = get_config(input_shape=33, n_hidden=5, patience=patience, dropout_rate=0.0)
    cfg.training.callbacks.EarlyStopping.start_from_epoch = start_from_epoch
    cfg.training.n_epochs = n_epochs
    logging.error("Start from epoch: %d", cfg.training.callbacks.EarlyStopping.start_from_epoch)
    logging.error("Patience: %d", cfg.training.callbacks.EarlyStopping.patience)

    ##################### 4. Split test data #####################
    # 1. Split 3 folds (60% – 30% – 10%)
    train_val_indices, test_indices = train_test_split(
        np.arange(features.shape[0]), test_size=0.1, random_state=seed, stratify=labels
    )
    before_train = {}
    after_train = {}
    kf = StratifiedKFold(n_splits=3, random_state=seed, shuffle=True)
    onehot_label = Preprocessing.one_hot_encoding_labels
    for i, (train_idx, val_idx) in enumerate(kf.split(train_val_indices, labels[train_val_indices])):
        train, val = train_val_indices[train_idx], train_val_indices[val_idx]
        test = test_indices
        
        # Convert to TensorFlow dataset object from numpy
        # feature: n x 33 ; label: n x 2
        train_ds = np_to_dataset(features[train], onehot_label(labels[train], 2), shuffle=True, seed=seed)
        val_ds = np_to_dataset(features[val], onehot_label(labels[val], 2), shuffle=False, seed=seed)
        test_ds = np_to_dataset(features[test], onehot_label(labels[test], 2), shuffle=False, seed=seed)
        
        # Load foundation model
        foundation_model = tf.keras.models.load_model(model_input)
        bf_val_perf = foundation_model.evaluate(val_ds)
        bf_test_perf = foundation_model.evaluate(test_ds)
        # Transfer learning 
        transfer_learning_model = train_model(
            train_ds, val_ds, cfg=cfg,
            folder=log_dir, 
            filename=f'fold_{i+1}',
            model_input=foundation_model, 
            seed=seed, 
        )
        
        # Evaluation
        af_val_perf = transfer_learning_model.evaluate(val_ds)
        af_test_perf = transfer_learning_model.evaluate(test_ds)
        
        bf_val_f1 = (bf_val_perf[3]*bf_val_perf[4]) / (bf_val_perf[3]*bf_val_perf[4]) 
        bf_test_f1 = (bf_test_perf[3]*bf_test_perf[4]) / (bf_test_perf[3]*bf_test_perf[4]) 
        before_train[i] = {
            'val_loss': bf_val_perf[0], 'val_accuracy': bf_val_perf[1], 'val_auc': bf_val_perf[2], 'val_precision': bf_val_perf[3], 'val_recall': bf_val_perf[4], 'val_f1': bf_val_f1,
            'test_loss': bf_test_perf[0], 'test_accuracy': bf_test_perf[1], 'test_auc': bf_test_perf[2], 'test_precision': bf_test_perf[3], 'test_recall': bf_test_perf[4], 'test_f1': bf_test_f1
        }
        after_train[i] = {
            'val_loss': af_val_perf[0], 'val_accuracy': af_val_perf[1], 'val_auc': af_val_perf[2], 'val_precision': af_val_perf[3], 'val_recall': af_val_perf[4], 'val_f1': af_val_perf[5],
            'test_loss': af_test_perf[0], 'test_accuracy': af_test_perf[1], 'test_auc': af_test_perf[2], 'test_precision': af_test_perf[3], 'test_recall': af_test_perf[4], 'test_f1': af_test_perf[5]
        }
        logging.error(
            f"Fold {i+1} before transfer:\n"
            f"  val_loss={bf_val_perf[0]:.3f}, val_acc={bf_val_perf[1]:.2%}, val_auc={bf_val_perf[2]:.2f}, val_precision={bf_val_perf[3]:.2f}, val_recall={bf_val_perf[4]:.2f}, val_f1={bf_val_f1:.2f}\n"
            f"  test_loss={bf_test_perf[0]:.3f}, test_acc={bf_test_perf[1]:.2%}, test_auc={bf_test_perf[2]:.2f}, test_precision={bf_test_perf[3]:.2f}, test_recall={bf_test_perf[4]:.2f}, test_f1={bf_test_f1:.2f}"
        )
        logging.error(
            f"Fold {i+1} after transfer:\n"
            f"  val_loss={af_val_perf[0]:.3f}, val_acc={af_val_perf[1]:.2%}, val_auc={af_val_perf[2]:.2f}, val_precision={af_val_perf[3]:.2f}, val_recall={af_val_perf[4]:.2f}, val_f1={af_val_perf[5]:.2f}\n"
            f"  test_loss={af_test_perf[0]:.3f}, test_acc={af_test_perf[1]:.2%}, test_auc={af_test_perf[2]:.2f}, test_precision={af_test_perf[3]:.2f}, test_recall={af_test_perf[4]:.2f}, test_f1={af_test_perf[5]:.2f}"
        )
    
    # Plot the loss/accuracy curve
    folds_history = [pd.read_csv(f'{log_dir}/history_fold_{i+1}.csv') for i in range(3)]
    fig = plot_acc_loss_3fold_CV(folds_history, 'Training History', name="")
    fig.savefig(f'{log_dir}/training_history.png')

    ###################### Save before and after training results ######################
    df_cols = ['val_loss', 'val_accuracy', 'val_auc', 'val_precision', 'val_recall', 'val_f1',
               'test_loss', 'test_accuracy', 'test_auc', 'test_precision', 'test_recall', 'test_f1']
    before_transfer = pd.DataFrame(columns=df_cols)
    after_transfer = pd.DataFrame(columns=df_cols)
    df_before_train = pd.DataFrame(before_train).T
    df_before_train.to_csv(f'{log_dir}/before_training.csv', index=False)
    before_transfer = pd.concat([before_transfer, df_before_train], axis=0)

    df_after_train = pd.DataFrame(after_train).T
    df_after_train.to_csv(f'{log_dir}/after_training.csv', index=False)
    after_transfer = pd.concat([after_transfer, df_after_train], axis=0)

    before_train_df = pd.DataFrame(before_train).T
    after_train_df = pd.DataFrame(after_train).T
    before_train_overall = pd.DataFrame(columns=before_train_df.columns)
    before_train_overall.loc['mean'] = before_train_df.mean()
    before_train_overall.loc['sem'] = before_train_df.sem()
    before_train_overall.to_csv(f'{log_dir}/before_training.csv', index=False)

    after_train_overall = pd.DataFrame(columns=after_train_df.columns)
    after_train_overall.loc['mean'] = after_train_df.mean()
    after_train_overall.loc['sem'] = after_train_df.sem()
    after_train_overall.to_csv(f'{log_dir}/after_training.csv', index=False)

    # Print out the results
    print_out = 'Before Training\n'
    print_out += '-----------------------------------------------------------------\n'
    print_out += 'val_loss\t%.2f±%.2f, val_accuracy\t%.2f±%.2f%%, \n' % (before_train_overall.loc['mean', 'val_loss'], before_train_overall.loc['sem', 'val_loss'], before_train_overall.loc['mean', 'val_accuracy']*100, before_train_overall.loc['sem', 'val_accuracy']*100)
    print_out += 'test_loss\t%.2f±%.2f, test_accuracy\t%.2f±%.2f%%, \n' % (before_train_overall.loc['mean', 'test_loss'], before_train_overall.loc['sem', 'test_loss'], before_train_overall.loc['mean', 'test_accuracy']*100, before_train_overall.loc['sem', 'test_accuracy']*100)
    print_out += '-----------------------------------------------------------------\n'
    print_out += 'After Training\n'
    print_out += 'val_loss\t%.2f±%.2f, val_accuracy\t%.2f±%.2f%%, \n' % (after_train_overall.loc['mean', 'val_loss'], after_train_overall.loc['sem', 'val_loss'], after_train_overall.loc['mean', 'val_accuracy']*100, after_train_overall.loc['sem', 'val_accuracy']*100)
    print_out += 'test_loss\t%.2f±%.2f, test_accuracy\t%.2f±%.2f%%, \n' % (after_train_overall.loc['mean', 'test_loss'], after_train_overall.loc['sem', 'test_loss'], after_train_overall.loc['mean', 'test_accuracy']*100, after_train_overall.loc['sem', 'test_accuracy']*100)
    print_out += '-----------------------------------------------------------------\n'
    logging.error(print_out) # Write to log
    logging.error("End Time = %s", datetime.datetime.now().strftime("%Y%m%d-%H%M")) # Write to log
    logging.error("#"*50) # Write to log