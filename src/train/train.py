import os
import datetime
import logging
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from .modules import Preprocessing, DelayedEarlyStopping, Callback_CSVLogger
from .modules import BinaryF1Score, GradientLoggingModel, GradientLogger
from .modules import build_model, np_to_dataset, plot_acc_loss, plot_acc_loss_3fold_CV, build_optimizer
from ..utils.settings import TANDEM_R20000, TANDEM_GJB2, TANDEM_RYR1, CLUSTER, ROOT_DIR
from ..utils.logger import LOGGER
from ..features import TANDEM_FEATS
from .config import get_config
from .process_data import getR20000, getTestset, onehot_encoding
from ..model.data_processing import probs2mode

filedir = os.path.dirname(os.path.abspath(__file__))

def get_seed(seed=150):
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    return seed

def use_all_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # Log GPUs using f-strings
    LOGGER.info(f"Num GPUs Available: {len(gpus)}")
    LOGGER.info(f"GPUs: {gpus}")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            LOGGER.error(f"Error setting memory_growth on GPUs: {e}")

def train_model(train_ds, val_ds, cfg, folder, filename, 
                model_input=None, seed=None, 
                initial_biase=None, logging_model=None,
                fold_num=None):
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
        model = build_model(cfg, initial_biase, logging_model=logging_model, verbose=False)
    
    optimizer = tf.keras.optimizers.Nadam(learning_rate=cfg.optimizer.learning_rate)
    optimizer.build(model.trainable_variables)
    
    if fold_num:
        # Calculate steps per epoch
        steps_per_epoch = len(train_ds)  # Adjust based on how you're generating batches
        cp_path = f'{folder}/checkpoints/model_{fold_num}/' + 'cp_{epoch:04d}.weights.h5'

        # Define ModelCheckpoint callback
        os.makedirs(f'{folder}/checkpoints/model_{fold_num}', exist_ok=True) # Ensure directory exists
        checkpoint_callback = ModelCheckpoint(
            filepath=cp_path,  # Save path format
            save_weights_only=True,  # Save only weights, not the entire model
            save_freq=20 * steps_per_epoch,  # Save every 20 epochs
            verbose=1  # Print a message when the model is saved
        )
    
    csv_logger = Callback_CSVLogger(
        data=[train_ds, val_ds],
        name=['train', 'val'],
        log_file=f'{folder}/history_{filename}.csv'
    )
    gradient_logger = GradientLogger(
        log_file=f'{folder}/gradient_{filename}.csv',
        W6_01_log_file=f'{folder}/W6_01_{filename}.csv',
        W6_02_log_file=f'{folder}/W6_02_{filename}.csv',
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
    if logging_model:
        callbacks = [early_stopping, csv_logger, gradient_logger]
    else:
        callbacks = [early_stopping, csv_logger]

    if fold_num:
        callbacks.extend([checkpoint_callback])
    
    model.fit(
        train_ds,
        epochs=cfg.training.n_epochs,
        validation_data=val_ds,
        callbacks=callbacks,
    )
    model.save(os.path.join(folder, f'model_{filename}.h5'), include_optimizer=True)
    
    if fold_num:
        # Save model weights
        model.save_weights(cp_path.format(epoch=0))
        
    return model
    
def reproduce_foundation_model(
        name='reproduce_foundation_model',
        featds=TANDEM_R20000,
        featset=TANDEM_FEATS['v1.1'],
        gjb2ds=TANDEM_GJB2,
        ryr1ds=TANDEM_RYR1,
        clstr=CLUSTER,
    ):
    """
    folds: dictionary contains information of 5 folds
    R20000: all SAVs (SAV_coords, features, labels) that are distributed into folds
    preprocess_feat: class object which helps to standardize new input (e.g., RYR1, GJB2)
    """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(ROOT_DIR, f'logs/{name}/{current_time}')
    logfile = os.path.join(log_dir, 'log.txt')
    os.makedirs(log_dir, exist_ok=True)
    LOGGER.start(logfile)
    LOGGER.info(f"Start Time = {current_time}")
    LOGGER.info(f"Tensorflow Version: {tf.__version__}") # Write to log    
    LOGGER.info("Tensorflow Version should be 2.17")
    LOGGER.info(f"Feature set: {featset}")
    seed=17
    patience = 50
    n_hidden = 5
    
    use_all_gpus()
    folds, R20000, preprocess_feat, df_clstr = getR20000(featds, clstr, feat_names=featset)
    # Save train data (train+val) for shap analysis
    R20000_train = np.vstack(
        (folds[1]['train']['x'],
        folds[1]['val']['x'])
    )
    np.save(f'{log_dir}/shap_background.npy', R20000_train)

    GJB2_knw, GJB2_unk = getTestset(gjb2ds, featset, preprocess_feat)
    RYR1_knw, RYR1_unk = getTestset(ryr1ds, featset, preprocess_feat)
    GJB2_notnan_SAV_coords, GJB2_notnan_labels, GJB2_notnan_features = GJB2_knw
    RYR1_notnan_SAV_coords, RYR1_notnan_labels, RYR1_notnan_features = RYR1_knw
    GJB2_nan_SAV_coords, GJB2_nan_labels, GJB2_nan_features = GJB2_unk
    RYR1_nan_SAV_coords, RYR1_nan_labels, RYR1_nan_features = RYR1_unk
    
    input_shape = R20000[2].shape[1]
    cfg = get_config(input_shape, n_hidden=n_hidden, patience=patience, dropout_rate=0.0)
    
    # Convert numpy array to tensorflow dataset
    GJB2_nan_ds = np_to_dataset(GJB2_nan_features, GJB2_nan_labels, shuffle=False, batch_size=cfg.training.batch_size, seed=seed)
    GJB2_notnan_ds = np_to_dataset(GJB2_notnan_features, GJB2_notnan_labels, shuffle=False, batch_size=cfg.training.batch_size, seed=seed)
    RYR1_nan_ds = np_to_dataset(RYR1_nan_features, RYR1_nan_labels, shuffle=False, batch_size=cfg.training.batch_size, seed=seed)
    RYR1_notnan_ds = np_to_dataset(RYR1_notnan_features, RYR1_notnan_labels, shuffle=False, batch_size=cfg.training.batch_size, seed=seed)

    ##################### 3. Train ####################################
    evaluations = {}
    models = []
    for i, fold in folds.items():
        train, val, test = fold['train'], fold['val'], fold['test']
        train_ds = np_to_dataset(train['x'], train['y'], shuffle=True, batch_size=cfg.training.batch_size, seed=seed)
        val_ds = np_to_dataset(val['x'], val['y'], shuffle=False, batch_size=cfg.training.batch_size, seed=seed)
        test_ds = np_to_dataset(test['x'], test['y'], shuffle=False, batch_size=cfg.training.batch_size, seed=seed)
        initial_biase = np.log([np.sum(train['y'][:, 0]) / np.sum(train['y'][:, 1])]) # Ref: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        model = train_model(
            train_ds, val_ds, cfg=cfg,
            folder=log_dir, filename=f'fold_{i+1}', model_input=None, 
            seed=seed, initial_biase=initial_biase, logging_model=False,
            # fold_num=i+1,
        )
        models.append(model)
        
        ### Evaluation
        val_rs = model.evaluate(val_ds, verbose=0)
        test_rs = model.evaluate(test_ds, verbose=0)
        GJB2_notnan_rs = model.evaluate(GJB2_notnan_ds, verbose=0)
        RYR1_notnan_rs = model.evaluate(RYR1_notnan_ds, verbose=0)

        msg = (
            f"Fold {i+1} - "
            f"val_loss: {val_rs[0]:.2f}, val_accuracy: {val_rs[1]*100:.1f}%, val_auc: {val_rs[2]:.2f}, "
            f"val_precision: {val_rs[3]:.2f}, val_recall: {val_rs[4]:.2f}, val_f1: {val_rs[5]:.2f}, "
            f"test_loss: {test_rs[0]:.2f}, test_accuracy: {test_rs[1]*100:.1f}%, test_auc: {test_rs[2]:.2f}, "
            f"test_precision: {test_rs[3]:.2f}, test_recall: {test_rs[4]:.2f}, test_f1: {test_rs[5]:.2f}, "
            f"RYR1_loss: {RYR1_notnan_rs[0]:.2f}, RYR1_accuracy: {RYR1_notnan_rs[1]*100:.1f}%, RYR1_auc: {RYR1_notnan_rs[2]:.2f}, "
            f"RYR1_precision: {RYR1_notnan_rs[3]:.2f}, RYR1_recall: {RYR1_notnan_rs[4]:.2f}, RYR1_notnan_f1: {RYR1_notnan_rs[5]:.2f}, "
            f"GJB2_loss: {GJB2_notnan_rs[0]:.2f}, GJB2_accuracy: {GJB2_notnan_rs[1]*100:.1f}%, GJB2_auc: {GJB2_notnan_rs[2]:.2f}, "
            f"GJB2_precision: {GJB2_notnan_rs[3]:.2f}, GJB2_recall: {GJB2_notnan_rs[4]:.2f}, GJB2_notnan_f1: {GJB2_notnan_rs[5]:.2f}"
        )
        LOGGER.info(msg)
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

    LOGGER.info("-----------------------------------------------------------------")
    LOGGER.info(f"Vali - loss: {df_overall.loc['mean', 'val_loss']:.2f}±{df_overall.loc['sem', 'val_loss']:.2f}, "
                f"accuracy: {df_overall.loc['mean', 'val_accuracy']*100:.1f}±{df_overall.loc['sem', 'val_accuracy']*100:.1f}%, "
                f"auc: {df_overall.loc['mean', 'val_auc']:.2f}±{df_overall.loc['sem', 'val_auc']:.2f}")
    LOGGER.info(f"Test - loss: {df_overall.loc['mean', 'test_loss']:.2f}±{df_overall.loc['sem', 'test_loss']:.2f}, "
                f"accuracy: {df_overall.loc['mean', 'test_accuracy']*100:.1f}±{df_overall.loc['sem', 'test_accuracy']*100:.1f}%, "
                f"auc: {df_overall.loc['mean', 'test_auc']:.2f}±{df_overall.loc['sem', 'test_auc']:.2f}")
    LOGGER.info(f"GJB2 - loss: {df_overall.loc['mean', 'GJB2_notnan_loss']:.2f}±{df_overall.loc['sem', 'GJB2_notnan_loss']:.2f}, "
                f"accuracy: {df_overall.loc['mean', 'GJB2_notnan_accuracy']*100:.1f}±{df_overall.loc['sem', 'GJB2_notnan_accuracy']*100:.1f}%, "
                f"auc: {df_overall.loc['mean', 'GJB2_notnan_auc']:.2f}±{df_overall.loc['sem', 'GJB2_notnan_auc']:.2f}")
    LOGGER.info(f"RYR1 - loss: {df_overall.loc['mean', 'RYR1_notnan_loss']:.2f}±{df_overall.loc['sem', 'RYR1_notnan_loss']:.2f}, "
                f"accuracy: {df_overall.loc['mean', 'RYR1_notnan_accuracy']*100:.1f}±{df_overall.loc['sem', 'RYR1_notnan_accuracy']*100:.1f}%, "
                f"auc: {df_overall.loc['mean', 'RYR1_notnan_auc']:.2f}±{df_overall.loc['sem', 'RYR1_notnan_auc']:.2f}")
    LOGGER.info("-----------------------------------------------------------------")
    # plot
    folds_history = [pd.read_csv(f'{log_dir}/history_fold_{i}.csv') for i in range(1, 6)]
    fig = plot_acc_loss(folds_history, 'Training History')
    fig.savefig(f'{log_dir}/training_history.png')

    # Make predictions on nan data
    # 0-4: 5 TANDEM models
    df_GJB2_nan = pd.DataFrame(columns=['SAV_coords', 0, 1, 2, 3, 4])
    df_GJB2_nan['SAV_coords'] = GJB2_nan_SAV_coords
    # Make prediction
    for i, model in enumerate(models):
        pred = model.predict(GJB2_nan_ds)
        pred = pred[:, 1] # Get the probability of class 1: pathogenic
        df_GJB2_nan[i] = pred
    df_GJB2_nan.to_csv(f'{log_dir}/GJB2_nan_pathogenicity_prob.csv', index=False)

    # Voting average
    GJB2_nan_probs = df_GJB2_nan[[0, 1, 2, 3, 4]].values
    np_GJB2_nan = probs2mode(GJB2_nan_probs)
    df_GJB2_nan = pd.DataFrame(np_GJB2_nan)
    df_GJB2_nan.to_csv(f'{log_dir}/GJB2_nan_pathogenicity_pred.csv', index=False)

    # Make predictions on nan data
    df_RYR1_nan = pd.DataFrame(columns=['SAV_coords', 0, 1, 2, 3, 4])
    df_RYR1_nan['SAV_coords'] = RYR1_nan_SAV_coords
    # Make prediction
    for i, model in enumerate(models):
        pred = model.predict(RYR1_nan_ds)
        pred = pred[:, 1] # Get the probability of class 1: pathogenic
        df_RYR1_nan[i] = pred
    df_RYR1_nan.to_csv(f'{log_dir}/RYR1_nan_pathogenicity_prob.csv', index=False)

    # Voting average
    RYR1_nan_probs = df_RYR1_nan[[0, 1, 2, 3, 4]].values
    np_RYR1_nan = probs2mode(RYR1_nan_probs)
    df_RYR1_nan = pd.DataFrame(np_RYR1_nan)
    df_RYR1_nan.to_csv(f'{log_dir}/RYR1_nan_pathogenicity_pred.csv', index=False)

def reproduce_transfer_learning_model(
        base_models,
        TANDEM_testSet,
        name,
        seed=73):

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(ROOT_DIR, 'logs', name, f'{current_time}-seed-{seed}')
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, 'log.txt')
    LOGGER.start(logfile)
    LOGGER.info(f"Start Time = {current_time}")
    use_all_gpus()

    # R20000_folds, R20000, preprocess_feat, test_knw, test_unk, input_shape = import_data(TANDEM_testSet)
    ##################### 1. Set up feature set #####################
    t_sel_feats = TANDEM_FEATS['v1.1']
    LOGGER.info(f"Feature set: {t_sel_feats}")
    R20000_folds, R20000, preprocess_feat = getR20000(TANDEM_R20000, CLUSTER, feat_names=t_sel_feats)
    test_knw, test_unk = getTestset(TANDEM_testSet, t_sel_feats, preprocess_feat) 

    SAV_coords, labels, features = test_knw
    VUS_coords, VUS_labels, VUS_features = test_unk
    labels = np.argmax(labels, axis=1)

    ##################### 3. Set up model configuration #####################
    patience = 50
    n_hidden = 5
    cfg = get_config(33, n_hidden=n_hidden, patience=patience, dropout_rate=0.0)
    cfg.training.callbacks.EarlyStopping.start_from_epoch = 10
    cfg.training.n_epochs = 1000
    LOGGER.info(f"Start from epoch: {cfg.training.callbacks.EarlyStopping.start_from_epoch}")

    ##################### 5. Split test data #####################
    # 1. Split 3 folds (60% – 30% – 10%)
    train_indices, test_indices = train_test_split(np.arange(len(labels)), test_size=0.1, random_state=seed, stratify=labels)
    # Save train data (train+val) for shap analysis
    testset_train = test_knw[2][train_indices]
    np.save(f'{log_dir}/shap_background.npy', testset_train)

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
        LOGGER.info(
            f"Fold {i+1} - "
            f"Train: {np.sum(labels[train])}pos + {len(train)-np.sum(labels[train])}neg, "
            f"Val: {np.sum(labels[val])}pos + {len(val)-np.sum(labels[val])}neg, "
            f"Test: {np.sum(labels[test])}pos + {len(test)-np.sum(labels[test])}neg"
        )
        LOGGER.info(f"Train: {SAV_coords[train]}")
        LOGGER.info(f"Val: {SAV_coords[val]}")
        LOGGER.info(f"Test: {SAV_coords[test]}")

    df_VUS_prob_list = []
    df_VUS_pred_list = []

    df_cols = ['R20000_val_loss', 'R20000_val_accuracy',  'R20000_val_auc', 'R20000_val_precision', 'R20000_val_recall', 'R20000_val_f1',
               'R20000_test_loss', 'R20000_test_accuracy', 'R20000_test_auc', 'R20000_test_precision', 'R20000_test_recall', 'R20000_test_f1',
               'val_loss', 'val_accuracy', 'val_auc', 'val_precision', 'val_recall', 'val_f1',
               'test_loss', 'test_accuracy', 'test_auc', 'test_precision', 'test_recall', 'test_f1',
               'knw_loss', 'knw_accuracy', 'knw_auc', 'knw_precision', 'knw_recall', 'knw_f1',
               ]
    
    baseline = pd.DataFrame(columns=df_cols)
    best = pd.DataFrame(columns=df_cols)

    before_transfer = pd.DataFrame(columns=df_cols)
    after_transfer = pd.DataFrame(columns=df_cols)

    fd_models = [os.path.join(base_models, f'model_fold_{i}.h5') for i in range(1, 6)]
    fd_models = [tf.keras.models.load_model(m) for m in fd_models]
    for model_idx, fd_model in enumerate(fd_models):
        model_dir = f'{log_dir}/model_{model_idx}' # 1 2 3 4 5
        os.makedirs(model_dir, exist_ok=True)
        R20000_fold = R20000_folds[model_idx]
        R20000_train, R20000_val, R20000_test = R20000_fold['train'], R20000_fold['val'], R20000_fold['test']
        R20000_val_ds = np_to_dataset(R20000_val['x'], R20000_val['y'], shuffle=False, batch_size=cfg.training.batch_size, seed=seed)
        R20000_test_ds = np_to_dataset(R20000_test['x'], R20000_test['y'], shuffle=False, batch_size=cfg.training.batch_size, seed=seed)
        
        VUS_ds = np_to_dataset(VUS_features, VUS_labels, shuffle=False, batch_size=cfg.training.batch_size)

        before_train = {}
        after_train = {}
        TEST_models = []
        for fold_idx in range(3):
            fold = folds[fold_idx]
            train, val, test = fold['train'], fold['val'], fold['test']
            x_train, y_train, SAVs_train = train['x'], train['y'], train['SAV_coords']
            x_val, y_val, SAVs_val = val['x'], val['y'], val['SAV_coords']
            x_test, y_test, SAVs_test  = test['x'], test['y'], test['SAV_coords']

            y_train = onehot_encoding(y_train, 2)
            y_val = onehot_encoding(y_val, 2)
            y_test = onehot_encoding(y_test, 2)

            train_ds = np_to_dataset(x_train, y_train, shuffle=True, batch_size=cfg.training.batch_size, seed=seed)
            val_ds = np_to_dataset(x_val, y_val, shuffle=False, batch_size=cfg.training.batch_size, seed=seed)
            test_ds = np_to_dataset(x_test, y_test, shuffle=False, batch_size=cfg.training.batch_size, seed=seed)

            y_knw = onehot_encoding(labels, 2)
            knw_ds  = np_to_dataset(features, y_knw, shuffle=False, batch_size=cfg.training.batch_size, seed=seed)
        
            ##################### 5. Train model on test data #####################
            fd_model_cp = tf.keras.models.clone_model(fd_model)
            fd_model_cp.set_weights(fd_model.get_weights())

            model = train_model(
                train_ds, 
                val_ds, 
                cfg=cfg,
                folder=model_dir, 
                filename=f'fold_{fold_idx+1}',
                model_input=fd_model_cp,
            )
            ### Evaluation before training
            before_R20000_val_performance = fd_model.evaluate(R20000_val_ds, verbose=0)
            before_R20000_test_performance = fd_model.evaluate(R20000_test_ds, verbose=0)
            before_val_performance = fd_model.evaluate(val_ds, verbose=0)
            before_test_performance = fd_model.evaluate(test_ds, verbose=0)
            before_knw_performance = fd_model.evaluate(knw_ds, verbose=0)
            ### Evaluation after training
            after_R20000_val_performance = model.evaluate(R20000_val_ds,verbose=0)
            after_R20000_test_performance = model.evaluate(R20000_test_ds,verbose=0)
            after_val_performance = model.evaluate(val_ds,verbose=0)
            after_test_performance = model.evaluate(test_ds,verbose=0)
            after_knw_performance = model.evaluate(knw_ds,verbose=0)

            before_train[fold_idx] = {
                'R20000_val_loss': before_R20000_val_performance[0], 'R20000_val_accuracy': before_R20000_val_performance[1], 'R20000_val_auc': before_R20000_val_performance[2], 'R20000_val_precision': before_R20000_val_performance[3], 'R20000_val_recall': before_R20000_val_performance[4],  'R20000_val_f1': before_R20000_val_performance[5], 
                'R20000_test_loss': before_R20000_test_performance[0], 'R20000_test_accuracy': before_R20000_test_performance[1], 'R20000_test_auc': before_R20000_test_performance[2], 'R20000_test_precision': before_R20000_test_performance[3], 'R20000_test_recall': before_R20000_test_performance[4], 'R20000_test_f1': before_R20000_test_performance[5],
                'val_loss': before_val_performance[0], 'val_accuracy': before_val_performance[1], 'val_auc': before_val_performance[2], 'val_precision': before_val_performance[3], 'val_recall': before_val_performance[4], 'val_f1': before_val_performance[5],
                'test_loss': before_test_performance[0], 'test_accuracy': before_test_performance[1], 'test_auc': before_test_performance[2], 'test_precision': before_test_performance[3], 'test_recall': before_test_performance[4], 'test_f1': before_test_performance[5],
                'knw_loss': before_knw_performance[0], 'knw_accuracy': before_knw_performance[1], 'knw_auc': before_knw_performance[2], 'knw_precision': before_knw_performance[3], 'knw_recall': before_knw_performance[4], 'knw_f1': before_knw_performance[5],
            }
            after_train[fold_idx] = {
                'R20000_val_loss': after_R20000_val_performance[0], 'R20000_val_accuracy': after_R20000_val_performance[1], 'R20000_val_auc': after_R20000_val_performance[2], 'R20000_val_precision': after_R20000_val_performance[3], 'R20000_val_recall': after_R20000_val_performance[4], 'R20000_val_f1': after_R20000_val_performance[5],
                'R20000_test_loss': after_R20000_test_performance[0], 'R20000_test_accuracy': after_R20000_test_performance[1], 'R20000_test_auc': after_R20000_test_performance[2], 'R20000_test_precision': after_R20000_test_performance[3], 'R20000_test_recall': after_R20000_test_performance[4], 'R20000_test_f1': after_R20000_test_performance[5],
                'val_loss': after_val_performance[0], 'val_accuracy': after_val_performance[1], 'val_auc': after_val_performance[2], 'val_precision': after_val_performance[3], 'val_recall': after_val_performance[4], 'val_f1': after_val_performance[5],
                'test_loss': after_test_performance[0], 'test_accuracy': after_test_performance[1], 'test_auc': after_test_performance[2], 'test_precision': after_test_performance[3], 'test_recall': after_test_performance[4], 'test_f1': after_test_performance[5],
                'knw_loss': after_knw_performance[0], 'knw_accuracy': after_knw_performance[1], 'knw_auc': after_knw_performance[2], 'knw_precision': after_knw_performance[3], 'knw_recall': after_knw_performance[4], 'knw_f1': after_knw_performance[5],
            }

            LOGGER.info(
                f"Fold {fold_idx+1} before - "
                f"R20000_val_loss: {before_R20000_val_performance[0]:.2f}, "
                f"R20000_val_accuracy: {before_R20000_val_performance[1]*100:.2f}%, "
                f"R20000_val_auc: {before_R20000_val_performance[2]:.2f}, "
                f"R20000_val_precision: {before_R20000_val_performance[3]:.2f}, "
                f"R20000_val_recall: {before_R20000_val_performance[4]:.2f}, "
                f"R20000_val_f1: {before_R20000_val_performance[5]:.2f}, "
                f"R20000_test_loss: {before_R20000_test_performance[0]:.2f}, "
                f"R20000_test_accuracy: {before_R20000_test_performance[1]*100:.2f}%, "
                f"R20000_test_auc: {before_R20000_test_performance[2]:.2f}, "
                f"R20000_test_precision: {before_R20000_test_performance[3]:.2f}, "
                f"R20000_test_recall: {before_R20000_test_performance[4]:.2f}, "
                f"R20000_test_f1: {before_R20000_test_performance[5]:.2f}, "
                f"val_loss: {before_val_performance[0]:.2f}, "
                f"val_accuracy: {before_val_performance[1]*100:.2f}%, "
                f"val_auc: {before_val_performance[2]:.2f}, "
                f"val_precision: {before_val_performance[3]:.2f}, "
                f"val_recall: {before_val_performance[4]:.2f}, "
                f"val_f1: {before_val_performance[5]:.2f}, "
                f"test_loss: {before_test_performance[0]:.2f}, "
                f"test_accuracy: {before_test_performance[1]*100:.2f}%, "
                f"test_auc: {before_test_performance[2]:.2f}, "
                f"test_precision: {before_test_performance[3]:.2f}, "
                f"test_recall: {before_test_performance[4]:.2f}, "
                f"test_f1: {before_test_performance[5]:.2f}, "
                f"knw_loss: {before_knw_performance[0]:.2f}, "
                f"knw_accuracy: {before_knw_performance[1]*100:.2f}%, "
                f"knw_auc: {before_knw_performance[2]:.2f}, "
                f"knw_precision: {before_knw_performance[3]:.2f}, "
                f"knw_recall: {before_knw_performance[4]:.2f}, "
                f"knw_f1: {before_knw_performance[5]:.2f}"
            )

            LOGGER.info(
                f"Fold {fold_idx+1} after - "
                f"R20000_val_loss: {after_R20000_val_performance[0]:.2f}, "
                f"R20000_val_accuracy: {after_R20000_val_performance[1]*100:.2f}%, "
                f"R20000_val_auc: {after_R20000_val_performance[2]:.2f}, "
                f"R20000_val_precision: {after_R20000_val_performance[3]:.2f}, "
                f"R20000_val_recall: {after_R20000_val_performance[4]:.2f}, "
                f"R20000_val_f1: {after_R20000_val_performance[5]:.2f}, "
                f"R20000_test_loss: {after_R20000_test_performance[0]:.2f}, "
                f"R20000_test_accuracy: {after_R20000_test_performance[1]*100:.2f}%, "
                f"R20000_test_auc: {after_R20000_test_performance[2]:.2f}, "
                f"R20000_test_precision: {after_R20000_test_performance[3]:.2f}, "
                f"R20000_test_recall: {after_R20000_test_performance[4]:.2f}, "
                f"R20000_test_f1: {after_R20000_test_performance[5]:.2f}, "
                f"val_loss: {after_val_performance[0]:.2f}, "
                f"val_accuracy: {after_val_performance[1]*100:.2f}%, "
                f"val_auc: {after_val_performance[2]:.2f}, "
                f"val_precision: {after_val_performance[3]:.2f}, "
                f"val_recall: {after_val_performance[4]:.2f}, "
                f"val_f1: {after_val_performance[5]:.2f}, "
                f"test_loss: {after_test_performance[0]:.2f}, "
                f"test_accuracy: {after_test_performance[1]*100:.2f}%, "
                f"test_auc: {after_test_performance[2]:.2f}, "
                f"test_precision: {after_test_performance[3]:.2f}, "
                f"test_recall: {after_test_performance[4]:.2f}, "
                f"test_f1: {after_test_performance[5]:.2f}, "
                f"knw_loss: {after_knw_performance[0]:.2f}, "
                f"knw_accuracy: {after_knw_performance[1]*100:.2f}%, "
                f"knw_auc: {after_knw_performance[2]:.2f}, "
                f"knw_precision: {after_knw_performance[3]:.2f}, "
                f"knw_recall: {after_knw_performance[4]:.2f}, "
                f"knw_f1: {after_knw_performance[5]:.2f}"
            )

            # Prediction test_ds
            preds = model.predict(test_ds)
            pathogenic_probs = preds[:, 1]
            predictions = np.argmax(preds, axis=1)
            test_labels = np.argmax(y_test, axis=1)

            LOGGER.info("Predictions on test data")
            for SAV, prob, pred, label in zip(SAVs_test, pathogenic_probs, predictions, test_labels):
                LOGGER.info(f"{SAV}\t{prob:.3f}\t{pred}\t{label}")

            # Validation set
            preds = model.predict(val_ds)
            pathogenic_probs = preds[:, 1]
            predictions = np.argmax(preds, axis=1)
            val_labels = np.argmax(y_val, axis=1)

            LOGGER.info("Predictions on val data")
            for SAV, prob, pred, label in zip(SAVs_val, pathogenic_probs, predictions, val_labels):
                LOGGER.info(f"{SAV}\t{prob:.3f}\t{pred}\t{label}")

            # Save the model
            model.save(f"{model_dir}/model_fold_{fold_idx+1}.h5")
            TEST_models.append(model)

        ####################### Make predictions on nan data #######################
        df_VUS = pd.DataFrame(columns=['SAV_coords', 0, 1, 2])
        df_VUS['SAV_coords'] = VUS_coords
        # df_VUS['SAV_coords'] = GJB2_nan_SAV_coords
        # Make prediction
        for idx, model in enumerate(TEST_models):
            pred = model.predict(VUS_ds)
            pred = pred[:, 1]
            df_VUS[idx] = pred
        df_VUS.to_csv(f'{model_dir}/VUS_pathogenicity_prob.csv', index=False)
        df_VUS_prob_list.append(df_VUS[[0, 1, 2]].copy()) # Save the probability of pathogenicity
        LOGGER.info(f"Save df_VUS to {f'{model_dir}/VUS_pathogenicity_prob.csv'}")

        for idx in range(3):
            df_VUS[idx] = df_VUS[idx].apply(lambda x: 1 if x > 0.5 else 0)
        df_VUS['final'] = df_VUS[[0, 1, 2]].mode(axis=1)[0]
        df_VUS.to_csv(f'{model_dir}/VUS_pathogenicity_pred.csv', index=False)
        df_VUS_pred_list.append(df_VUS[[0, 1, 2]].copy())
        LOGGER.info(f"Save df_VUS to {f'{model_dir}/VUS_pathogenicity_pred.csv'}")

        # Plot training history
        folds_history = [pd.read_csv(f'{model_dir}/history_fold_{j}.csv') for j in range(1, 4)]
        fig = plot_acc_loss_3fold_CV(folds_history, 'Training History', name="")
        fig.savefig(f'{model_dir}/training_history.png')
        LOGGER.info(f"Save folds_history to {f'{model_dir}/training_history.png'}")

        ###################### Save before and after training results ######################
        df_before_train = pd.DataFrame(before_train).T
        df_before_train.to_csv(f'{model_dir}/before_training.csv', index=False)
        before_transfer = pd.concat([before_transfer, df_before_train], axis=0)

        df_after_train = pd.DataFrame(after_train).T
        df_after_train.to_csv(f'{model_dir}/after_training.csv', index=False)
        after_transfer = pd.concat([after_transfer, df_after_train], axis=0)

        before_train_df = pd.DataFrame(before_train).T
        after_train_df = pd.DataFrame(after_train).T
        before_train_overall = pd.DataFrame(columns=before_train_df.columns)
        before_train_overall.loc['mean'] = before_train_df.mean()
        before_train_overall.loc['std'] = before_train_df.std()
        before_train_overall.loc['sem'] = before_train_df.sem()
        before_train_overall.to_csv(f'{model_dir}/before_training.csv', index=False)
        LOGGER.info(f"Save before_train_overall to {f'{model_dir}/before_training.csv'}")

        after_train_overall = pd.DataFrame(columns=after_train_df.columns)
        after_train_overall.loc['mean'] = after_train_df.mean()
        after_train_overall.loc['std'] = after_train_df.std()
        after_train_overall.loc['sem'] = after_train_df.sem()
        after_train_overall.to_csv(f'{model_dir}/after_training.csv', index=False)
        LOGGER.info(f"Save after_train_overall to {f'{model_dir}/after_training.csv'}")

        # Print out the results
        print_out = 'Before Training\n'
        print_out += '-----------------------------------------------------------------\n'
        print_out += 'R20000_val_loss\t%.2f±%.2f, R20000_val_accuracy\t%.2f±%.2f%%, \n' % (before_train_overall.loc['mean', 'R20000_val_loss'], before_train_overall.loc['sem', 'R20000_val_loss'], before_train_overall.loc['mean', 'R20000_val_accuracy']*100, before_train_overall.loc['sem', 'R20000_val_accuracy']*100)
        print_out += 'R20000_test_loss\t%.2f±%.2f, R20000_test_accuracy\t%.2f±%.2f%%, \n' % (before_train_overall.loc['mean', 'R20000_test_loss'], before_train_overall.loc['sem', 'R20000_test_loss'], before_train_overall.loc['mean', 'R20000_test_accuracy']*100, before_train_overall.loc['sem', 'R20000_test_accuracy']*100)
        print_out += 'val_loss\t%.2f±%.2f, val_accuracy\t%.2f±%.2f%%, \n' % (before_train_overall.loc['mean', 'val_loss'], before_train_overall.loc['sem', 'val_loss'], before_train_overall.loc['mean', 'val_accuracy']*100, before_train_overall.loc['sem', 'val_accuracy']*100)
        print_out += 'test_loss\t%.2f±%.2f, test_accuracy\t%.2f±%.2f%%, \n' % (before_train_overall.loc['mean', 'test_loss'], before_train_overall.loc['sem', 'test_loss'], before_train_overall.loc['mean', 'test_accuracy']*100, before_train_overall.loc['sem', 'test_accuracy']*100)
        print_out += 'knw_loss\t%.2f±%.2f, knw_accuracy\t%.2f±%.2f%%, \n' % (before_train_overall.loc['mean', 'knw_loss'], before_train_overall.loc['sem', 'knw_loss'], before_train_overall.loc['mean', 'knw_accuracy']*100, before_train_overall.loc['sem', 'knw_accuracy']*100)
        print_out += '-----------------------------------------------------------------\n'
        print_out += 'After Training\n'
        print_out += 'R20000_val_loss\t%.2f±%.2f, R20000_val_accuracy\t%.2f±%.2f%%, \n' % (after_train_overall.loc['mean', 'R20000_val_loss'], after_train_overall.loc['sem', 'R20000_val_loss'], after_train_overall.loc['mean', 'R20000_val_accuracy']*100, after_train_overall.loc['sem', 'R20000_val_accuracy']*100)
        print_out += 'R20000_test_loss\t%.2f±%.2f, R20000_test_accuracy\t%.2f±%.2f%%, \n' % (after_train_overall.loc['mean', 'R20000_test_loss'], after_train_overall.loc['sem', 'R20000_test_loss'], after_train_overall.loc['mean', 'R20000_test_accuracy']*100, after_train_overall.loc['sem', 'R20000_test_accuracy']*100)
        print_out += 'val_loss\t%.2f±%.2f, val_accuracy\t%.2f±%.2f%%, \n' % (after_train_overall.loc['mean', 'val_loss'], after_train_overall.loc['sem', 'val_loss'], after_train_overall.loc['mean', 'val_accuracy']*100, after_train_overall.loc['sem', 'val_accuracy']*100)
        print_out += 'test_loss\t%.2f±%.2f, test_accuracy\t%.2f±%.2f%%, \n' % (after_train_overall.loc['mean', 'test_loss'], after_train_overall.loc['sem', 'test_loss'], after_train_overall.loc['mean', 'test_accuracy']*100, after_train_overall.loc['sem', 'test_accuracy']*100)
        print_out += 'knw_loss\t%.2f±%.2f, knw_accuracy\t%.2f±%.2f%%, \n' % (after_train_overall.loc['mean', 'knw_loss'], after_train_overall.loc['sem', 'knw_loss'], after_train_overall.loc['mean', 'knw_accuracy']*100, after_train_overall.loc['sem', 'knw_accuracy']*100)
        print_out += '-----------------------------------------------------------------\n'
        LOGGER.info(print_out) # Write to log

        baseline.loc[f'mean_{model_idx}'] = before_train_overall.loc['mean']
        baseline.loc[f'std_{model_idx}'] = before_train_overall.loc['std']
        baseline.loc[f'sem_{model_idx}'] = before_train_overall.loc['sem']
        best.loc[f'mean_{model_idx}'] = after_train_overall.loc['mean']
        best.loc[f'std_{model_idx}'] = after_train_overall.loc['std']
        best.loc[f'sem_{model_idx}'] = after_train_overall.loc['sem']

    # Concatenate all predictions
    df_VUS_prob = pd.concat(df_VUS_prob_list, axis=1)
    df_VUS_pred = pd.concat(df_VUS_pred_list, axis=1)
    # Add SAV_coords at first column
    df_VUS_prob.insert(0, 'SAV_coords', VUS_coords)
    df_VUS_pred.insert(0, 'SAV_coords', VUS_coords)

    # Rename columns
    df_VUS_prob.columns = ['SAV_coords'] + list(range(15))
    df_VUS_pred.columns = ['SAV_coords'] + list(range(15))

    # Mode predictions of 15 models:
    df_VUS_pred['final'] = df_VUS_pred[[i for i in range(15)]].mode(axis=1)[0]
    df_VUS_pred['ratio'] = df_VUS_pred[[i for i in range(15)]].apply(lambda x: x.value_counts().max()/x.value_counts().sum(), axis=1)
    # Average of one decision as probability: Only the model gives that decision
    # Make decision using voting by using 15 models (do not be biased by models)

    preds = df_VUS_pred['final'].values
    for i, pred in enumerate(preds):
        # np.where from df_prob.iloc[i, 1:] == pred
        # Take the average of the probabilities
        # print the average
        pred_probs = df_VUS_prob.iloc[i, 1:]
        pred_probs = pred_probs[df_VUS_pred.iloc[i, 1:-2] == pred]
        prob = pred_probs.mean()
        prob_sem = pred_probs.sem()
        df_VUS_pred.loc[i, 'final_prob'] = prob
        df_VUS_pred.loc[i, 'final_prob_sem'] = prob_sem

    df_VUS_prob.to_csv(f'{log_dir}/VUS_pathogenicity_prob_total.csv', index=False)
    df_VUS_pred.to_csv(f'{log_dir}/VUS_pathogenicity_pred_total.csv', index=False)

    baseline.to_csv(f'{log_dir}/baseline.csv', index=False)
    best.to_csv(f'{log_dir}/best.csv', index=False)

    before_transfer.to_csv(f'{log_dir}/before_transfer.csv', index=False)
    after_transfer.to_csv(f'{log_dir}/after_transfer.csv', index=False)

    LOGGER.info(f"Save df_VUS_prob to {f'{log_dir}/VUS_pathogenicity_prob_total.csv'}")
    LOGGER.info(f"Save df_VUS_pred to {f'{log_dir}/VUS_pathogenicity_pred_total.csv'}")
    LOGGER.info(f"Save baseline to {f'{log_dir}/baseline.csv'}")
    LOGGER.info(f"Save best to {f'{log_dir}/csv.csv'}")
    LOGGER.info(f"Save before_transfer to {f'{log_dir}/before_transfer.csv'}")
    LOGGER.info(f"Save after_transfer to {f'{log_dir}/after_transfer.csv'}")
    LOGGER.info(f"End Time = {datetime.datetime.now().strftime('%Y%m%d-%H%M')}") # Write to log
    LOGGER.info("#"*50) # Write to log
