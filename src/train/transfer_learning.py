import os 
import logging
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from .run import use_all_gpus, get_config
from .run import getR20000, getTestset
from ..utils.settings import FEAT_STATS, dynamics_feat, structure_feat, seq_feat
from ..utils.settings import TANDEM_R20000, TANDEM_GJB2, TANDEM_RYR1, TANDEM_PKD1
from ..utils.settings import ROOT_DIR, CLUSTER
from .modules import np_to_dataset, Preprocessing, plot_acc_loss_3fold_CV, Callback_CSVLogger, DelayedEarlyStopping, build_optimizer, BinaryF1Score
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
    return folds, R20000, preprocess_feat, test_knw, test_unk, input_shape

def train_model(base_models,
                TANDEM_testSet,
                name,
                seed=73):
    NAME_OF_EXPERIMENT = f'TransferLearning_{name}'
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(ROOT_DIR, 'logs', NAME_OF_EXPERIMENT, f'{name}-{current_time}-seed-{seed}')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=f'{log_dir}/log.txt', level=logging.ERROR, format='%(message)s')
    logging.error("Start Time = %s", current_time)
    use_all_gpus()

    R20000_folds, R20000, preprocess_feat, test_knw, test_unk, input_shape = import_data(TANDEM_testSet)

    SAV_coords, labels, features = test_knw
    VUS_coords, VUS_labels, VUS_features = test_unk
    labels = np.argmax(labels, axis=1)

    ##################### 3. Set up model configuration #####################
    patience = 50
    n_hidden = 5
    cfg = get_config(input_shape, n_hidden=n_hidden, patience=patience, dropout_rate=0.0)
    cfg.training.callbacks.EarlyStopping.start_from_epoch = 10
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
    for model_idx in range(5):
        model_dir = f'{log_dir}/model_{model_idx}'
        os.makedirs(model_dir, exist_ok=True)
        
        R20000_fold = R20000_folds[model_idx]
        R20000_train, R20000_val, R20000_test = R20000_fold['train'], R20000_fold['val'], R20000_fold['test']
        R20000_train_ds = np_to_dataset(R20000_train['x'], R20000_train['y'], shuffle=True, batch_size=cfg.training.batch_size, seed=seed)
        R20000_val_ds = np_to_dataset(R20000_val['x'], R20000_val['y'], shuffle=False, batch_size=cfg.training.batch_size, seed=seed)
        R20000_test_ds = np_to_dataset(R20000_test['x'], R20000_test['y'], shuffle=False, batch_size=cfg.training.batch_size, seed=seed)

        VUS_ds = np_to_dataset(VUS_features, VUS_labels, shuffle=False, batch_size=cfg.training.batch_size)

        logging.error("=.= No. %d", model_idx) # Write to log
        logging.error('-----------------------------------------------------------------')
        before_train = {}
        after_train = {}
        TEST_models = []
        for split_idx in range(3):
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
                log_file=f'{model_dir}/history_fold_{split_idx}.csv',
            )
            early_stopping = DelayedEarlyStopping(**cfg.training.callbacks.EarlyStopping)
            models = [os.path.join(base_models, f'model_fold_{i}.h5') for i in range(1, 6)]
            model = tf.keras.models.load_model(models[model_idx])
            
            # Optimizer
            optimizer = build_optimizer(cfg)
            model.compile(optimizer=optimizer, loss=cfg.training.loss, 
                metrics=['accuracy', 
                        tf.keras.metrics.AUC(name='auc'), 
                        tf.keras.metrics.Precision(name='precision'), 
                        tf.keras.metrics.Recall(name='recall'),
                        BinaryF1Score(name='f1_score')
                    ]
            )
            
            ### Evaluation before training
            before_R20000_val_performance = model.evaluate(R20000_val_ds)
            before_R20000_test_performance = model.evaluate(R20000_test_ds)
            before_val_performance = model.evaluate(val_ds)
            before_test_performance = model.evaluate(test_ds)
            # Add evaluation of testset before split
            before_knw_performance = model.evaluate(knw_ds)

            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=cfg.training.n_epochs,
                callbacks=[csv_logger, early_stopping],
                verbose=1,
                batch_size=cfg.training.batch_size,
            )

            ### Evaluation after training
            after_R20000_val_performance = model.evaluate(R20000_val_ds)
            after_R20000_test_performance = model.evaluate(R20000_test_ds)
            after_val_performance = model.evaluate(val_ds)
            after_test_performance = model.evaluate(test_ds)
            # Add evaluation of testset after split
            after_knw_performance = model.evaluate(knw_ds)

            before_train[split_idx] = {
                'R20000_val_loss': before_R20000_val_performance[0], 'R20000_val_accuracy': before_R20000_val_performance[1], 'R20000_val_auc': before_R20000_val_performance[2], 'R20000_val_precision': before_R20000_val_performance[3], 'R20000_val_recall': before_R20000_val_performance[4],  'R20000_val_f1': before_R20000_val_performance[5], 
                'R20000_test_loss': before_R20000_test_performance[0], 'R20000_test_accuracy': before_R20000_test_performance[1], 'R20000_test_auc': before_R20000_test_performance[2], 'R20000_test_precision': before_R20000_test_performance[3], 'R20000_test_recall': before_R20000_test_performance[4], 'R20000_test_f1': before_R20000_test_performance[5],
                'val_loss': before_val_performance[0], 'val_accuracy': before_val_performance[1], 'val_auc': before_val_performance[2], 'val_precision': before_val_performance[3], 'val_recall': before_val_performance[4], 'val_f1': before_val_performance[5],
                'test_loss': before_test_performance[0], 'test_accuracy': before_test_performance[1], 'test_auc': before_test_performance[2], 'test_precision': before_test_performance[3], 'test_recall': before_test_performance[4], 'test_f1': before_test_performance[5],
                'knw_loss': before_knw_performance[0], 'knw_accuracy': before_knw_performance[1], 'knw_auc': before_knw_performance[2], 'knw_precision': before_knw_performance[3], 'knw_recall': before_knw_performance[4], 'knw_f1': before_knw_performance[5],
            }
            after_train[split_idx] = {
                'R20000_val_loss': after_R20000_val_performance[0], 'R20000_val_accuracy': after_R20000_val_performance[1], 'R20000_val_auc': after_R20000_val_performance[2], 'R20000_val_precision': after_R20000_val_performance[3], 'R20000_val_recall': after_R20000_val_performance[4], 'R20000_val_f1': after_R20000_val_performance[5],
                'R20000_test_loss': after_R20000_test_performance[0], 'R20000_test_accuracy': after_R20000_test_performance[1], 'R20000_test_auc': after_R20000_test_performance[2], 'R20000_test_precision': after_R20000_test_performance[3], 'R20000_test_recall': after_R20000_test_performance[4], 'R20000_test_f1': after_R20000_test_performance[5],
                'val_loss': after_val_performance[0], 'val_accuracy': after_val_performance[1], 'val_auc': after_val_performance[2], 'val_precision': after_val_performance[3], 'val_recall': after_val_performance[4], 'val_f1': after_val_performance[5],
                'test_loss': after_test_performance[0], 'test_accuracy': after_test_performance[1], 'test_auc': after_test_performance[2], 'test_precision': after_test_performance[3], 'test_recall': after_test_performance[4], 'test_f1': after_test_performance[5],
                'knw_loss': after_knw_performance[0], 'knw_accuracy': after_knw_performance[1], 'knw_auc': after_knw_performance[2], 'knw_precision': after_knw_performance[3], 'knw_recall': after_knw_performance[4], 'knw_f1': after_knw_performance[5],
            }
            # logging.error("Fold %d before - R20000_val_loss: %.2f, R20000_val_accuracy: %.2f%%, R20000_val_auc: %.2f, R20000_val_precision: %.2f, R20000_val_recall: %.2f, R20000_test_loss: %.2f, R20000_test_accuracy: %.2f%%, R20000_test_auc: %.2f, R20000_test_precision: %.2f, R20000_test_recall: %.2f, val_loss: %.2f, val_accuracy: %.2f%%, val_auc: %.2f, val_precision: %.2f, val_recall: %.2f, test_loss: %.2f, test_accuracy: %.2f%%, test_auc: %.2f, test_precision: %.2f, test_recall: %.2f, knw_loss: %.2f, knw_accuracy: %.2f%%, knw_auc: %.2f, knw_precision: %.2f, knw_recall: %.2f",
            #               split_idx+1, 
            #               before_R20000_val_performance[0], before_R20000_val_performance[1]*100, before_R20000_val_performance[2], before_R20000_val_performance[3], before_R20000_val_performance[4],
            #               before_R20000_test_performance[0], before_R20000_test_performance[1]*100, before_R20000_test_performance[2], before_R20000_test_performance[3], before_R20000_test_performance[4],
            #               before_val_performance[0], before_val_performance[1]*100, before_val_performance[2], before_val_performance[3], before_val_performance[4],
            #               before_test_performance[0], before_test_performance[1]*100, before_test_performance[2], before_test_performance[3], before_test_performance[4],
            #               before_knw_performance[0], before_knw_performance[1]*100, before_knw_performance[2], before_knw_performance[3], before_knw_performance[4])
            # logging.error("Fold %d after - R20000_val_loss: %.2f, R20000_val_accuracy: %.2f%%, R20000_val_auc: %.2f, R20000_val_precision: %.2f, R20000_val_recall: %.2f, R20000_test_loss: %.2f, R20000_test_accuracy: %.2f%%, R20000_test_auc: %.2f, R20000_test_precision: %.2f, R20000_test_recall: %.2f, val_loss: %.2f, val_accuracy: %.2f%%, val_auc: %.2f, val_precision: %.2f, val_recall: %.2f, test_loss: %.2f, test_accuracy: %.2f%%, test_auc: %.2f, test_precision: %.2f, test_recall: %.2f, knw_loss: %.2f, knw_accuracy: %.2f%%, knw_auc: %.2f, knw_precision: %.2f, knw_recall: %.2f",
            #               split_idx+1,
            #               after_R20000_val_performance[0], after_R20000_val_performance[1]*100, after_R20000_val_performance[2], after_R20000_val_performance[3], after_R20000_val_performance[4],
            #               after_R20000_test_performance[0], after_R20000_test_performance[1]*100, after_R20000_test_performance[2], after_R20000_test_performance[3], after_R20000_test_performance[4],
            #               after_val_performance[0], after_val_performance[1]*100, after_val_performance[2], after_val_performance[3], after_val_performance[4],
            #               after_test_performance[0], after_test_performance[1]*100, after_test_performance[2], after_test_performance[3], after_test_performance[4],
            #               after_knw_performance[0], after_knw_performance[1]*100, after_knw_performance[2], after_knw_performance[3], after_knw_performance[4])
            
            logging.error("Fold %d before - R20000_val_loss: %.2f, R20000_val_accuracy: %.2f%%, R20000_val_auc: %.2f, R20000_val_precision: %.2f, R20000_val_recall: %.2f, R20000_val_f1: %.2f, R20000_test_loss: %.2f, R20000_test_accuracy: %.2f%%, R20000_test_auc: %.2f, R20000_test_precision: %.2f, R20000_test_recall: %.2f, R20000_test_f1: %.2f, val_loss: %.2f, val_accuracy: %.2f%%, val_auc: %.2f, val_precision: %.2f, val_recall: %.2f, val_f1: %.2f, test_loss: %.2f, test_accuracy: %.2f%%, test_auc: %.2f, test_precision: %.2f, test_recall: %.2f, test_f1: %.2f, knw_loss: %.2f, knw_accuracy: %.2f%%, knw_auc: %.2f, knw_precision: %.2f, knw_recall: %.2f, knw_f1: %.2f",
              split_idx+1, 
              before_R20000_val_performance[0], before_R20000_val_performance[1]*100, before_R20000_val_performance[2], before_R20000_val_performance[3], before_R20000_val_performance[4], before_R20000_val_performance[5],
              before_R20000_test_performance[0], before_R20000_test_performance[1]*100, before_R20000_test_performance[2], before_R20000_test_performance[3], before_R20000_test_performance[4], before_R20000_test_performance[5],
              before_val_performance[0], before_val_performance[1]*100, before_val_performance[2], before_val_performance[3], before_val_performance[4], before_val_performance[5],
              before_test_performance[0], before_test_performance[1]*100, before_test_performance[2], before_test_performance[3], before_test_performance[4], before_test_performance[5],
              before_knw_performance[0], before_knw_performance[1]*100, before_knw_performance[2], before_knw_performance[3], before_knw_performance[4], before_knw_performance[5])
            logging.error("Fold %d after - R20000_val_loss: %.2f, R20000_val_accuracy: %.2f%%, R20000_val_auc: %.2f, R20000_val_precision: %.2f, R20000_val_recall: %.2f, R20000_val_f1: %.2f, R20000_test_loss: %.2f, R20000_test_accuracy: %.2f%%, R20000_test_auc: %.2f, R20000_test_precision: %.2f, R20000_test_recall: %.2f, R20000_test_f1: %.2f, val_loss: %.2f, val_accuracy: %.2f%%, val_auc: %.2f, val_precision: %.2f, val_recall: %.2f, val_f1: %.2f, test_loss: %.2f, test_accuracy: %.2f%%, test_auc: %.2f, test_precision: %.2f, test_recall: %.2f, test_f1: %.2f, knw_loss: %.2f, knw_accuracy: %.2f%%, knw_auc: %.2f, knw_precision: %.2f, knw_recall: %.2f, knw_f1: %.2f",
                split_idx+1,
                after_R20000_val_performance[0], after_R20000_val_performance[1]*100, after_R20000_val_performance[2], after_R20000_val_performance[3], after_R20000_val_performance[4], after_R20000_val_performance[5],
                after_R20000_test_performance[0], after_R20000_test_performance[1]*100, after_R20000_test_performance[2], after_R20000_test_performance[3], after_R20000_test_performance[4], after_R20000_test_performance[5],
                after_val_performance[0], after_val_performance[1]*100, after_val_performance[2], after_val_performance[3], after_val_performance[4], after_val_performance[5],
                after_test_performance[0], after_test_performance[1]*100, after_test_performance[2], after_test_performance[3], after_test_performance[4], after_test_performance[5],
                after_knw_performance[0], after_knw_performance[1]*100, after_knw_performance[2], after_knw_performance[3], after_knw_performance[4], after_knw_performance[5])
            # Prediction test_ds
            preds = model.predict(test_ds)
            pathogenic_probs = preds[:, 1]
            predictions = np.argmax(preds, axis=1)
            test_labels = np.argmax(y_test, axis=1)
            # Print out the predictions: SAVs_test, pathogenic_probs, predictions
            logging.error("Predictions on test data")
            for SAV, prob, pred, label in zip(SAVs_test, pathogenic_probs, predictions, test_labels):
                logging.error("%s\t%.3f\t%d\t%d", SAV, prob, pred, label)
            
            preds = model.predict(val_ds)
            pathogenic_probs = preds[:, 1]
            predictions = np.argmax(preds, axis=1)
            val_labels = np.argmax(y_val, axis=1)
            logging.error("Predictions on val data")
            # Print out the predictions: SAVs_val, pathogenic_probs, predictions
            for SAV, prob, pred, label in zip(SAVs_val, pathogenic_probs, predictions, val_labels):
                logging.error("%s\t%.3f\t%d\t%d", SAV, prob, pred, label)
            # logging.error(f"Fold {split_idx+1} - R20000_val_loss: {after_R20000_val_performance[0]:.2f}, R20000_val_accuracy: {after_R20000_val_performance[1]*100:.2f}%, R20000_test_loss: {after_R20000_test_performance[0]:.2f}, R20000_test_accuracy: {after_R20000_test_performance[1]*100:.2f}%")
            # logging.error(f"Fold {split_idx+1} - val_loss: {after_val_performance[0]:.2f}, val_accuracy: {after_val_performance[1]*100:.2f}%, test_loss: {after_test_performance[0]:.2f}, test_accuracy: {after_test_performance[1]*100:.2f}%")

            model.save(f'{model_dir}/model_fold_{split_idx+1}.h5')
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

        for idx in range(3):
            df_VUS[idx] = df_VUS[idx].apply(lambda x: 1 if x > 0.5 else 0)
        df_VUS['final'] = df_VUS[[0, 1, 2]].mode(axis=1)[0]
        df_VUS.to_csv(f'{model_dir}/VUS_pathogenicity_pred.csv', index=False)
        df_VUS_pred_list.append(df_VUS[[0, 1, 2]].copy())

        # Plot training history
        folds_history = [pd.read_csv(f'{model_dir}/history_fold_{j}.csv') for j in range(3)]
        fig = plot_acc_loss_3fold_CV(folds_history, 'Training History', name="")
        fig.savefig(f'{model_dir}/training_history.png')

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

        after_train_overall = pd.DataFrame(columns=after_train_df.columns)
        after_train_overall.loc['mean'] = after_train_df.mean()
        after_train_overall.loc['std'] = after_train_df.std()
        after_train_overall.loc['sem'] = after_train_df.sem()
        after_train_overall.to_csv(f'{model_dir}/after_training.csv', index=False)

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
        logging.error(print_out) # Write to log

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

    logging.error("End Time = %s", datetime.datetime.now().strftime("%Y%m%d-%H%M")) # Write to log
    logging.error("#"*50) # Write to log
