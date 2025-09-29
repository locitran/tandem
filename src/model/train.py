import numpy as np
import random
import os 

from dataclasses import dataclass
import tensorflow as tf
from keras.saving import register_keras_serializable

def get_seed(seed=0):
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    return seed

class DelayedEarlyStopping(tf.keras.callbacks.EarlyStopping):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, 
                 verbose=0, mode='auto', baseline=None, 
                 restore_best_weights=False, start_from_epoch=50):
        super(DelayedEarlyStopping, self).__init__(
            monitor=monitor, min_delta=min_delta, patience=patience, 
            verbose=verbose, mode=mode, baseline=baseline, 
            restore_best_weights=restore_best_weights)
        self.patience = patience
        self.best_weights = None
        self.start_from_epoch = start_from_epoch
        self.best_epoch = 0  # Initialize the best_epoch attribute

    def on_train_begin(self, logs=None):
        # Initialize variables at the beginning of training
        self.wait = 0  # Number of epochs waited after the best epoch
        self.stopped_epoch = 0  # The epoch where training stops
        self.best = np.inf  # Initialize the best as infinity

    def on_epoch_end(self, epoch, logs=None):
        # Only start early stopping after a certain number of epochs
        if epoch > self.start_from_epoch:
            current = logs.get("val_loss")
            if np.less(current, self.best):
                self.best = current
                self.wait = 0
                self.best_epoch = epoch
                # Record the best weights if the current result is better (less loss)
                self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    self.model.set_weights(self.best_weights)

@register_keras_serializable(package="custom", name="BinaryF1Score")
class BinaryF1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(BinaryF1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)  # assumes sigmoid output
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()
            
# Create customized callbacks to record test accuracy and loss
class Callback_CSVLogger(tf.keras.callbacks.Callback):
    def __init__(self, data, name, log_file):
        super(Callback_CSVLogger, self).__init__()
        self.data = data
        self.name = name
        self.log_file = log_file

    # tensorflow 1.17
    def on_train_begin(self, logs=None):
        self.metric_names = [] # [<Mean name=loss>, <CompileMetrics name=compile_metrics>]
        self.metric_names.append(self.model.metrics[0].name)
        for metric in self.model.metrics[1]._user_metrics:
            # self.metric_names.append(metric.name)
            if isinstance(metric, str):
                self.metric_names.append(metric)
            else:
                self.metric_names.append(metric.name)

    def on_epoch_end(self, epoch, logs=None):
        metrics_names = self.metric_names
        header = 'epoch,'
        for name in self.name:
            for metric in metrics_names:
                header += f"{name}_{metric},"
        header = header[:-1] + '\n'
        if epoch == 0:
            with open(self.log_file, 'w') as f:
                f.write(header)
        
        row = f"{epoch+1},"
        for ds in self.data:
            ds_logs = self.model.evaluate(ds, verbose=0)
            ds_logs = dict(zip(metrics_names, ds_logs))
            for metric in metrics_names:
                row += f"{ds_logs[metric]},"
        row = row[:-1] + '\n'
        with open(self.log_file, 'a') as f:
            f.write(row)

def _compile_model(model, lr):
    optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
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
    return model

def train_model(
    train_ds, 
    val_ds, 
    cfg, 
    folder, 
    filename, 
    model_input=None, 
):
    # Set seed for each fold => Every fold has the same weight initialization
    # Reproducibility
    get_seed(cfg.seed)
    model = model_input
    csv_logger = Callback_CSVLogger(
        data=[train_ds, val_ds],
        name=['train', 'val'],
        log_file=f'{folder}/history_{filename}.csv'
    )
    early_stopping = DelayedEarlyStopping(
        monitor=cfg.monitor,
        patience=cfg.patience,
        mode=cfg.monitor_mode,
        restore_best_weights=cfg.restore_best_weights,
        start_from_epoch=cfg.start_from_epoch,
        verbose=cfg.verbose,
    )
    model = _compile_model(model, cfg.learning_rate)
    callbacks = [early_stopping, csv_logger]
    model.fit(
        train_ds,
        epochs=cfg.n_epochs,
        validation_data=val_ds,
        callbacks=callbacks,
    )
    model.save(os.path.join(folder, f'model_{filename}.h5'), include_optimizer=True)
    return model

@dataclass
class TLConfig:
    # optimization
    learning_rate: float = 5e-5
    batch_size: int = 300
    n_epochs: int = 300
    patience: int = 50
    restore_best_weights: bool = True
    start_from_epoch: int = 10
    # data / CV
    val_splits: int = 3    # Stratified KFold
    test_size: float = 0.10
    seed: int = 42
    # checkpoint metric
    monitor: str = "val_loss"     # requires AUC metric
    monitor_mode: str = "min"
    # saving
    # save_format: Literal["keras", "h5", "savedmodel"] = "keras"  # Keras v3 prefers .keras
    save_best_only: bool = True
    # logging
    verbose: int = 1