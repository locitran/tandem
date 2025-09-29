import os
import datetime
import logging
import pandas as pd
import numpy as np
import random
from .modules import Preprocessing, DelayedEarlyStopping, Callback_CSVLogger, BinaryF1Score
from .modules import build_model, np_to_dataset, build_optimizer, plot_acc_loss, build_model_from_config
from .split_data import split_data
from .config import model_config
from tensorflow.keras.callbacks import ModelCheckpoint

_LOGGER = logging.getLogger(__name__)
import tensorflow as tf
_LOGGER.error("Tensorflow Version: %s", tf.__version__) # Write to log    

def use_all_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # Log gpus
    _LOGGER.error("Num GPUs Available: %d", len(gpus)) # Write to log
    _LOGGER.error("GPUs: %s", gpus) # Write to log
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            _LOGGER.error(e) # Write to log

def get_seed(seed=150):
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    return seed

def getR20000(feat_path, clstr_path, feat_names, folder=None):
    # Load data
    df = pd.read_csv(feat_path)
    # Split data into 5 folds
    folds = split_data(feat_path, clstr_path)
    
    if folder:
        os.makedirs(folder, exist_ok=True)
        plot_label_ratio(folds, folder)
    _LOGGER.error("Load R20000 dataset")
    
    SAV_coords = df['SAV_coords'].values
    features = df[feat_names].values
    preprocess_feat = Preprocessing(features)
    features = preprocess_feat.fill_na_mean(features)
    features = preprocess_feat.normalize(features)
    labels = df['labels'].values
    labels = Preprocessing.one_hot_encoding_labels(labels, 2)

    entities = ['train', 'val', 'test']
    for i, fold in folds.items():
        for entity in entities:
            data = fold[entity]
            data_SAV_coords = data['SAV_coords']
            idx = df[df['SAV_coords'].isin(data_SAV_coords)].index
            folds[i][entity]['x'] = features[idx]
            folds[i][entity]['y'] = labels[idx]

    R20000 = [SAV_coords, labels, features]

    # Add Sep 6 to try "Several potential swappable clusters"
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
    return folds, R20000, preprocess_feat

def getTestset(feat_path, feat_names, preprocess_feat, name=None):
    df = pd.read_csv(feat_path)

    _LOGGER.error('*'*50)
    _LOGGER.error('Missing values in the dataframe:')
    for i, feat_name in enumerate(df.columns):
        if df[feat_name].isnull().sum() > 0:
            _LOGGER.error('%s: \t\t %d' % (feat_name, df[feat_name].isnull().sum()))

    SAV_coords = df['SAV_coords'].values
    features = df[feat_names].values
    features = preprocess_feat.fill_na_mean(features)
    features = preprocess_feat.normalize(features)
    labels = df['labels'].values # Contains NaN values

    nan_check = np.isnan(labels)
    nan_SAV_coords = SAV_coords[nan_check]
    nan_labels = labels[nan_check]
    nan_features = features[nan_check]

    notnan_SAV_coords = SAV_coords[~nan_check]
    notnan_labels = labels[~nan_check]
    notnan_labels = Preprocessing.one_hot_encoding_labels(notnan_labels, 2)
    notnan_features = features[~nan_check]

    n_benign = np.sum(notnan_labels[:, 0])
    n_pathogenic = np.sum(notnan_labels[:, 1])
    name = name if name else 'Unknown'
    _LOGGER.error("No. %s SAVs %d (benign), %d (pathogenic), and %d (NaN)", name, n_benign, n_pathogenic, nan_features.shape[0])

    knw = [notnan_SAV_coords, notnan_labels, notnan_features]
    unk = [nan_SAV_coords, nan_labels, nan_features]
    return knw, unk

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
    _LOGGER.error("Label ratio plot saved to %s", out) # Write to log

def get_config(input_shape=33, n_hidden=5, patience=50, dropout_rate=0., 
               n_neuron_per_hidden=None, n_neuron_last_hidden=None, verbose=True):
    cfg = model_config()
    cfg.model.input.n_neurons = input_shape
    _LOGGER.error("Input Layer: %d", cfg.model.input.n_neurons) # Write to log
    cfg['model']['input']['dropout_rate'] = dropout_rate

    # No. of neurons in the output layer
    n_neuron_per_hidden = input_shape if n_neuron_per_hidden is None else n_neuron_per_hidden
    n_neuron_last_hidden = 10 if n_neuron_last_hidden is None else n_neuron_last_hidden
    
    weight_initialization = ['glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform']
    initializer = 'glorot_uniform'
    
    for item in cfg.model.hidden:
        del cfg['model']['hidden'][item]
    for i in range(n_hidden):
        n_neuron = n_neuron_last_hidden if i == n_hidden - 1 else n_neuron_per_hidden
        hidden_name = f'hidden_{i:02d}'
        cfg['model']['hidden'][hidden_name] = {
            'activation': 'gelu',
            'batch_norm': False,
            'dropout_rate': dropout_rate,
            'initializer': initializer,
            'l1': 0,
            'l2': 0.0001,
            'n_neurons': n_neuron
        }
    cfg['training']['callbacks']['EarlyStopping']['patience'] = patience

    if verbose:
        from prettytable import PrettyTable
        # print cfg as table
        tb = PrettyTable()
        tb.field_names = ["Layer", "Activation", "Batch Norm", "Dropout Rate", "Initializer", "L1", "L2", "N Neurons"]
        tb.add_row(["Input", "-", "-", cfg.model.input.dropout_rate, "-", "-", "-", cfg.model.input.n_neurons])
        for layer in cfg.model.hidden:
            tb.add_row([layer, cfg.model.hidden[layer].activation, cfg.model.hidden[layer].batch_norm, cfg.model.hidden[layer].dropout_rate, cfg.model.hidden[layer].initializer, cfg.model.hidden[layer].l1, cfg.model.hidden[layer].l2, cfg.model.hidden[layer].n_neurons])
        tb.add_row(["Output", cfg.model.output.activation, "-", "-", "-", "-", "-", cfg.model.output.n_neurons])
        _LOGGER.error("Model Configuration: \n%s", tb) # Write to log

        # print training and optimizer
        tb = PrettyTable()
        tb.field_names = ["Training", "Batch Size", "N Epochs", "Loss", "Metrics"]
        tb.add_row(["Training", cfg.training.batch_size, cfg.training.n_epochs, cfg.training.loss, cfg.training.metrics])
        tb.add_row(["Optimizer", cfg.optimizer.learning_rate, cfg.optimizer.name, "-", "-"])
        _LOGGER.error("Training Configuration: \n%s", tb) # Write to log
    return cfg

def train_model(folds, cfg, log_dir, 
                GJB2_knw, GJB2_unk,
                RYR1_knw, RYR1_unk, seed=None):

    GJB2_notnan_SAV_coords, GJB2_notnan_labels, GJB2_notnan_features = GJB2_knw
    RYR1_notnan_SAV_coords, RYR1_notnan_labels, RYR1_notnan_features = RYR1_knw
    GJB2_nan_SAV_coords, GJB2_nan_labels, GJB2_nan_features = GJB2_unk
    RYR1_nan_SAV_coords, RYR1_nan_labels, RYR1_nan_features = RYR1_unk
    models = []
    
    # np_to_ds GJB2 and RYR1
    GJB2_nan_ds = np_to_dataset(GJB2_nan_features, GJB2_nan_labels, shuffle=False, batch_size=cfg.training.batch_size, seed=seed)
    GJB2_notnan_ds = np_to_dataset(GJB2_notnan_features, GJB2_notnan_labels, shuffle=False, batch_size=cfg.training.batch_size, seed=seed)
    RYR1_nan_ds = np_to_dataset(RYR1_nan_features, RYR1_nan_labels, shuffle=False, batch_size=cfg.training.batch_size, seed=seed)
    RYR1_notnan_ds = np_to_dataset(RYR1_notnan_features, RYR1_notnan_labels, shuffle=False, batch_size=cfg.training.batch_size, seed=seed)

    evaluations = {}
    for i, fold in folds.items():
        train, val, test = fold['train'], fold['val'], fold['test']
        train_ds = np_to_dataset(train['x'], train['y'], shuffle=True, batch_size=cfg.training.batch_size, seed=seed)
        val_ds = np_to_dataset(val['x'], val['y'], shuffle=False, batch_size=cfg.training.batch_size, seed=seed)
        test_ds = np_to_dataset(test['x'], test['y'], shuffle=False, batch_size=cfg.training.batch_size, seed=seed)
        initial_biase = np.log([np.sum(train['y'][:, 0]) / np.sum(train['y'][:, 1])]) # Ref: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        
        # Set seed for each fold => Every fold has the same weight initialization
        seed = get_seed(seed)
        # _LOGGER.error("Seed: %d", seed) if i == 0 else None

        # Build model
        model = build_model(cfg, initial_biase)
        # model = build_model_from_config(cfg, initial_biase)
        # if i == 0:
        #     _LOGGER.error("Model Summary for Fold %d:", i+1)
        #     input = tf.random.normal((1, cfg['model']['input']['n_neurons']))
        #     output = model(input)
        #     model.summary(print_fn=lambda x: _LOGGER.error(x))
        # _LOGGER.error(f"Number of parameters: {model.count_params()}") if i == 0 else None

        ### Setting
        # Callbacks
        csv_logger = Callback_CSVLogger(
            data=[train_ds, val_ds], # test_ds, GJB2_notnan_ds, RYR1_notnan_ds],
            name=['train', 'val'],#, 'test', 'GJB2', 'RYR1'],
            log_file=f'{log_dir}/history_fold_{i+1}.csv'
        )
        early_stopping = DelayedEarlyStopping(**cfg.training.callbacks.EarlyStopping)

        # Calculate steps per epoch
        steps_per_epoch = len(train_ds)  # Adjust based on how you're generating batches
        cp_path = f'{log_dir}/checkpoints/model_{i+1}/' + 'cp_{epoch:04d}.weights.h5'

        # Define ModelCheckpoint callback
        os.makedirs(f'{log_dir}/checkpoints/model_{i+1}', exist_ok=True) # Ensure directory exists
        checkpoint_callback = ModelCheckpoint(
            filepath=cp_path,  # Save path format
            save_weights_only=True,  # Save only weights, not the entire model
            save_freq=40 * steps_per_epoch,  # Save every 20 epochs
            verbose=1  # Print a message when the model is saved
        )

        # Optimizer
        optimizer = build_optimizer(cfg)
        # compile more metrics: accuracy, auc, f1-score, precision, recall
        model.compile(optimizer=optimizer, loss=cfg.training.loss, 
                      metrics=['accuracy', 
                               tf.keras.metrics.AUC(name='auc'), 
                               tf.keras.metrics.Precision(name='precision'), 
                               tf.keras.metrics.Recall(name='recall'),
                               BinaryF1Score(name='f1_score')
                            ]
                )

        # Save model weights
        model.save_weights(cp_path.format(epoch=0))

        # Class weights
        class_weight = cfg.training.class_weight
        class_weight = folds[i]['train']['class_weight'] if class_weight == 'auto' else None

        ### Training
        model.fit(
            train_ds,
            epochs=cfg.training.n_epochs,
            validation_data=val_ds,
            class_weight=class_weight,
            callbacks=[early_stopping, csv_logger, checkpoint_callback],
            batch_size=cfg.training.batch_size,
        )

        # Save the final model weights for the best epoch
        model.save_weights(cp_path.format(epoch=early_stopping.best_epoch))        
        model.save_weights(cp_path.format(epoch=early_stopping.stopped_epoch))

        ### Evaluation
        val_rs = model.evaluate(val_ds)
        test_rs = model.evaluate(test_ds)
        GJB2_notnan_rs = model.evaluate(GJB2_notnan_ds)
        RYR1_notnan_rs = model.evaluate(RYR1_notnan_ds)

        evaluations[i] = {
            'val_loss': val_rs[0], 'val_accuracy': val_rs[1], 'val_auc': val_rs[2], 'val_precision': val_rs[3], 'val_recall': val_rs[4], 'val_f1': val_rs[5],
            'test_loss': test_rs[0], 'test_accuracy': test_rs[1], 'test_auc': test_rs[2], 'test_precision': test_rs[3], 'test_recall': test_rs[4], 'test_f1': test_rs[5],
            'GJB2_notnan_loss': GJB2_notnan_rs[0], 'GJB2_notnan_accuracy': GJB2_notnan_rs[1], 'GJB2_notnan_auc': GJB2_notnan_rs[2], 'GJB2_notnan_precision': GJB2_notnan_rs[3], 'GJB2_notnan_recall': GJB2_notnan_rs[4], 'GJB2_notnan_f1': GJB2_notnan_rs[5],
            'RYR1_notnan_loss': RYR1_notnan_rs[0], 'RYR1_notnan_accuracy': RYR1_notnan_rs[1], 'RYR1_notnan_auc': RYR1_notnan_rs[2], 'RYR1_notnan_precision': RYR1_notnan_rs[3], 'RYR1_notnan_recall': RYR1_notnan_rs[4], 'RYR1_notnan_f1': RYR1_notnan_rs[5],
        }
        msg = "Fold %d - val_loss: %.2f, val_accuracy: %.1f%%, val_auc: %.2f, val_precision: %.2f, val_recall: %.2f, val_f1: %.2f, " + \
                "test_loss: %.2f, test_accuracy: %.1f%%, test_auc: %.2f, test_precision: %.2f, test_recall: %.2f, test_f1: %.2f, " + \
                "RYR1_loss: %.2f, RYR1_accuracy: %.1f%%, RYR1_auc: %.2f, RYR1_precision: %.2f, RYR1_recall: %.2f, GJB2_notnan_f1: %.2f, " + \
                "GJB2_loss: %.2f, GJB2_accuracy: %.1f%%, GJB2_auc: %.2f, GJB2_precision: %.2f, GJB2_recall: %.2f, RYR1_notnan_f1: %.2f"
        _LOGGER.error(msg, i+1, val_rs[0], val_rs[1] * 100, val_rs[2], val_rs[3], val_rs[4], val_rs[5],
                                test_rs[0], test_rs[1] * 100, test_rs[2], test_rs[3], test_rs[4], test_rs[5], 
                                RYR1_notnan_rs[0], RYR1_notnan_rs[1] * 100, RYR1_notnan_rs[2], RYR1_notnan_rs[3], RYR1_notnan_rs[4], GJB2_notnan_rs[5],
                                GJB2_notnan_rs[0], GJB2_notnan_rs[1] * 100, GJB2_notnan_rs[2], GJB2_notnan_rs[3], GJB2_notnan_rs[4], RYR1_notnan_rs[5])
        model.save(f'{log_dir}/model_fold_{i+1}.h5')
        # model.save(f'{log_dir}/model_fold_{i+1}')
        models.append(model)

    df_evaluations = pd.DataFrame(evaluations).T
    df_evaluations.to_csv(f'{log_dir}/evaluations.csv', index=False)

    # df_overall: mean row, std row, sem row
    df_overall = pd.DataFrame(columns=df_evaluations.columns)
    df_overall.loc['mean'] = df_evaluations.mean()
    df_overall.loc['std'] = df_evaluations.std()
    df_overall.loc['sem'] = df_evaluations.sem()
    df_overall.to_csv(f'{log_dir}/overall.csv', index=False)

    _LOGGER.error("-----------------------------------------------------------------")
    _LOGGER.error("Vali - loss: %.2f±%.2f, accuracy: %.1f±%.1f%%, auc: %.2f±%.2f", df_overall.loc['mean', 'val_loss'], df_overall.loc['sem', 'val_loss'], df_overall.loc['mean', 'val_accuracy'] * 100, df_overall.loc['sem', 'val_accuracy'] * 100, df_overall.loc['mean', 'val_auc'], df_overall.loc['sem', 'val_auc'])
    _LOGGER.error("Test - loss: %.2f±%.2f, accuracy: %.1f±%.1f%%, auc: %.2f±%.2f", df_overall.loc['mean', 'test_loss'], df_overall.loc['sem', 'test_loss'], df_overall.loc['mean', 'test_accuracy'] * 100, df_overall.loc['sem', 'test_accuracy'] * 100, df_overall.loc['mean', 'test_auc'], df_overall.loc['sem', 'test_auc'])
    _LOGGER.error("GJB2 - loss: %.2f±%.2f, accuracy: %.1f±%.1f%%, auc: %.2f±%.2f", df_overall.loc['mean', 'GJB2_notnan_loss'], df_overall.loc['sem', 'GJB2_notnan_loss'], df_overall.loc['mean', 'GJB2_notnan_accuracy'] * 100, df_overall.loc['sem', 'GJB2_notnan_accuracy'] * 100, df_overall.loc['mean', 'GJB2_notnan_auc'], df_overall.loc['sem', 'GJB2_notnan_auc'])
    _LOGGER.error("RYR1 - loss: %.2f±%.2f, accuracy: %.1f±%.1f%%, auc: %.2f±%.2f", df_overall.loc['mean', 'RYR1_notnan_loss'], df_overall.loc['sem', 'RYR1_notnan_loss'], df_overall.loc['mean', 'RYR1_notnan_accuracy'] * 100, df_overall.loc['sem', 'RYR1_notnan_accuracy'] * 100, df_overall.loc['mean', 'RYR1_notnan_auc'], df_overall.loc['sem', 'RYR1_notnan_auc'])
    _LOGGER.error("-----------------------------------------------------------------")

    # Make predictions on nan data
    df_GJB2_nan = pd.DataFrame(columns=['SAV_coords', 0, 1, 2, 3, 4])
    df_GJB2_nan['SAV_coords'] = GJB2_nan_SAV_coords
    # Make prediction
    for i, model in enumerate(models):
        pred = model.predict(GJB2_nan_ds)
        pred = pred[:, 1] # Get the probability of class 1: pathogenic
        df_GJB2_nan[i] = pred
    df_GJB2_nan.to_csv(f'{log_dir}/GJB2_nan_pathogenicity_prob.csv', index=False)

    for i in range(5):
        df_GJB2_nan[i] = df_GJB2_nan[i].apply(lambda x: 1 if x > 0.5 else 0)
    df_GJB2_nan['final'] = df_GJB2_nan[[0, 1, 2, 3, 4]].mode(axis=1)[0]
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

    for i in range(5):
        df_RYR1_nan[i] = df_RYR1_nan[i].apply(lambda x: 1 if x > 0.5 else 0)
    df_RYR1_nan['final'] = df_RYR1_nan[[0, 1, 2, 3, 4]].mode(axis=1)[0]
    df_RYR1_nan.to_csv(f'{log_dir}/RYR1_nan_pathogenicity_pred.csv', index=False)

    # plot
    folds_history = [pd.read_csv(f'{log_dir}/history_fold_{i}.csv') for i in range(1, 6)]
    fig = plot_acc_loss(folds_history, 'Training History')
    fig.savefig(f'{log_dir}/training_history.png')
    return models, df_overall

