import os
import csv
import tensorflow as tf
import pandas as pd
import numpy as np

from dataclasses import asdict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from .features import TANDEM_FEATS
from .features.features import Features
from .utils.settings import TANDEM_v1dot1, TANDEM_R20000
from .utils.logger import LOGGER
from .model.data_processing import Preprocessing, onehot_encoding, probs2mode, np2ds
from .model.train import TLConfig, train_model

class Tandem(Features):
    
    def __init__(self, query, refresh=False, **kwargs):
        super().__init__(query, refresh, **kwargs)
        self.setR20000()
        self.models = self.setModels()
    #### -------- Calculate predictions ------- #####

    def setModels(self, folder=TANDEM_v1dot1):
        """Import models from the given folder.
        Args:
            folder (str): Folder containing the models.
        Returns:
            models (list): List of models.
        """
        assert os.path.exists(folder), f"Folder {folder} does not exist."
        LOGGER.info(f"{folder}")
        models = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.h5'):
                    models.append(os.path.join(root, file))

        assert len(models) > 0, f"No models found in {folder}."
        LOGGER.info(f"Found {len(models)} models in {folder}.")
        LOGGER.info(f"Loading models from {folder}.")
        models = [tf.keras.models.load_model(model) for model in models]
        return models

    def setR20000(self, data=TANDEM_R20000):
        df = pd.read_csv(data)
        fm = df[TANDEM_FEATS['v1.1']].values
        self.preprocess = Preprocessing(fm)

    def _calcPredictions(self, models=None):
        assert self.featMatrix is not None, 'Feature matrix not set.'
        # Convert the feature matrix to a NumPy array
        # fm = self.featMatrix.view(np.float64).reshape(self.nSAVs, -1)
        feat_names = self.featMatrix.dtype.names
        fm = np.column_stack([self.featMatrix[name] for name in feat_names])
        fm = self.featMatrix
        fm = self.preprocess(fm)
        
        # Load foundation models
        fd_models = self.models
        probs = []
        for model in fd_models:
            pred = model.predict(fm)
            pred = pred[:, 1] # Get the probability of class 1: pathogenic
            probs.append(pred)
        probs = np.column_stack(probs)
        self.data['tandem'] = probs2mode(probs)

        # Load transfer learning models
        if models:
            tf_models = self.setModels(models)
            probs = []
            for model in tf_models:
                pred = model.predict(fm)
                pred = pred[:, 1] # Get the probability of class 1: pathogenic
                probs.append(pred)
            probs = np.array(probs).reshape(self.nSAVs, -1)
            self.data['tandem_tf'] = probs2mode(probs)

    def getPredictions(self, models=None, folder='.', filename=None):
        # calc predictions
        self._calcPredictions(models)

        # Convert to df
        df_tandem = pd.DataFrame(self.data['tandem'].tolist(), columns=self.data['tandem'].dtype.names)
        df_tandem_tf = pd.DataFrame(self.data['tandem_tf'].tolist(), columns=self.data['tandem_tf'].dtype.names) if models else None
        # if tandem_tf exists, add its columns with suffix "_tf"
        if df_tandem_tf is not None:
            # rename TF columns to have _tf suffix
            df_tandem_tf = df_tandem_tf.add_suffix('_tf')
            # merge by index (row-aligned)
            df_tandem = pd.concat([df_tandem, df_tandem_tf], axis=1)
        # finally add SAVs column
        df_tandem.insert(loc=0, column='SAVs', value=self.data["SAVs"])

        if filename:
            sav_w = 15
            vote_w = 8
            path_w = 15
            decision_w = 15
            filepath = os.path.join(folder, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                # header parts for base columns
                header_parts = [
                    f'{"SAV":<{sav_w}}', # " | ",
                    f'{"vote":<{vote_w}}',
                    f'{"probability":<{path_w}}',
                    f'{"decision":<{decision_w}}'
                ]
                # add tf columns if they exist
                if any(c.endswith("_tf") for c in df_tandem.columns):
                    header_parts += [#" | ",
                        f'{"vote_tf":<{vote_w}}',
                        f'{"probability_tf":<{path_w}}',
                        f'{"decision_tf":<{decision_w}}'
                    ]
                header = "".join(header_parts) + "\n"
                f.write(header)

                # rows
                for _, data in df_tandem.iterrows():
                    sav = data["SAVs"]
                    # base values
                    vote = data.get("ratio")
                    vote_s = f"{float(vote):.3f}" if pd.notnull(vote) else ""
                    path = data.get("path_probs")
                    sem = data.get("path_probs_sem")
                    if pd.notnull(path) and pd.notnull(sem):
                        path_s = f"{float(path):.3f}±{float(sem):.3f}"
                    elif pd.notnull(path):
                        path_s = f"{float(path):.3f}"
                    else:
                        path_s = ""
                    decision = str(data.get("decision", "") or "")

                    line_parts = [ # build line parts
                        f"{sav:<{sav_w}}", # " | ",
                        f"{vote_s:<{vote_w}}",
                        f"{path_s:<{path_w}}",
                        f"{decision:<{decision_w}}",
                    ]
                    
                    if "ratio_tf" in data: # add tf values if present
                        vote_tf = data.get("ratio_tf")
                        vote_tf_s = f"{float(vote_tf):.3f}" if pd.notnull(vote_tf) else ""
                    
                        path_tf = data.get("path_probs_tf")
                        sem_tf = data.get("path_probs_sem_tf")
                        if pd.notnull(path_tf) and pd.notnull(sem_tf):
                            path_tf_s = f"{float(path_tf):.3f}±{float(sem_tf):.3f}"
                        elif pd.notnull(path_tf):
                            path_tf_s = f"{float(path_tf):.3f}"
                        else:
                            path_tf_s = ""
                        decision_tf = str(data.get("decision_tf", "") or "")
                    
                        line_parts += [ #" | ",
                            f"{vote_tf_s:<{vote_w}}",
                            f"{path_tf_s:<{path_w}}",
                            f"{decision_tf:<{decision_w}}",
                        ]
                    # join and finish line
                    line = "".join(line_parts) + "\n"
                    f.write(line)
                LOGGER.info(f'Predictions are saved to {filepath}')
        return df_tandem

    #### -------- Transfer learning ------- #####

    def setConfig(self, config=None):
        default = TLConfig()
        cfg = asdict(default)
        if config:
            cfg.update({k: v for k, v in config.items() if k in cfg})
        self.config = TLConfig(**cfg)

    def history_avg(self, history):
        metrics = ["loss", "accuracy", "auc", "precision", "recall", "f1"]
        summary = {}
        for model_type in ["fd", "tf"]:
            summary[model_type] = {}
            for split in ["val", "test"]:
                arr = np.array(history[model_type][split])  # shape (n_runs, n_metrics)
                means = arr.mean(axis=0)
                stds  = arr.std(axis=0, ddof=1)
                sems  = stds / np.sqrt(arr.shape[0])  # SEM
                mins  = arr.min(axis=0)
                maxs  = arr.max(axis=0)

                summary[model_type][split] = {
                    m: {
                        "mean": float(mu),
                        "std": float(sd),
                        "sem": float(se),
                        "min": float(mn),
                        "max": float(mx)
                    }
                    for m, mu, se, sd, mn, mx in zip(metrics, means, sems, stds, mins, maxs)
                }
        # --- Print: Foundation models ---
        fd_title = "Foundation models"
        LOGGER.info(fd_title)
        LOGGER.info(f"{'val':>15}{'std':>6}{'sem':>6}{'min':>6}{'max':>6}"
                    f"{'test':>9}{'std':>5}{'sem':>6}{'min':>6}{'max':>6}")

        for metric in metrics:
            v = summary["fd"]["val"][metric]
            t = summary["fd"]["test"][metric]
            line = (f"{metric:>10}: "
                    f"{v['mean']:.3f} {v['std']:.3f} {v['sem']:.3f} {v['min']:.3f} {v['max']:.3f}   "
                    f"{t['mean']:.3f} {t['std']:.3f} {t['sem']:.3f} {t['min']:.3f} {t['max']:.3f}")
            LOGGER.info(line)

        # --- Print: Transfer learning models ---
        tf_title = "Transfer learning models"
        LOGGER.info(tf_title)
        LOGGER.info(f"{'val':>15}{'std':>6}{'sem':>6}{'min':>6}{'max':>6}"
                    f"{'test':>9}{'std':>5}{'sem':>6}{'min':>6}{'max':>6}")

        for metric in metrics:
            v = summary["tf"]["val"][metric]
            t = summary["tf"]["test"][metric]
            line = (f"{metric:>10}: "
                    f"{v['mean']:.3f} {v['std']:.3f} {v['sem']:.3f} {v['min']:.3f} {v['max']:.3f}   "
                    f"{t['mean']:.3f} {t['std']:.3f} {t['sem']:.3f} {t['min']:.3f} {t['max']:.3f}")
            LOGGER.info(line)

    def history_to_csv(self, history, filename="history.csv"):
        """
        Save training history of foundation vs transfer models into CSV format.

        history: dict
            {
                'fd': {'val': [...], 'test': [...]},
                'tf': {'val': [...], 'test': [...]}
            }
        filename: str
            Output CSV file name
        """
        metrics = ["loss", "accuracy", "auc", "precision", "recall", "f1"]
        filepath = os.path.join(self.options['job_directory'], filename)
        with open(filepath, mode="w", newline="") as f:
            writer = csv.writer(f)
            # header
            writer.writerow([
                "fold", "model_type", "split", *metrics
            ])
            n_folds = len(history["fd"]["val"])
            for fold in range(n_folds):
                # foundation model
                writer.writerow([fold+1, "foundation", "val",  *history['fd']['val'][fold]])
                writer.writerow([fold+1, "foundation", "test", *history['fd']['test'][fold]])
                # transfer model
                writer.writerow([fold+1, "transfer", "val",  *history['tf']['val'][fold]])
                writer.writerow([fold+1, "transfer", "test", *history['tf']['test'][fold]])
        LOGGER.info(f"[INFO] History saved to {filepath}")

    def train(self, name, filename):
        assert self.featMatrix is not None, "Feature matrix not set."
        assert self._isColSet("labels"), "Labels not set."
        assert self.config is not None, "Config not set."
        LOGGER.timeit('_train')
        job_dir = self.options['job_directory']

        cfg = self.config
        feat_names = self.featMatrix.dtype.names
        X = np.column_stack([self.featMatrix[name] for name in feat_names])
        X = self.preprocess(X)  # ensure scaler
        y = np.asarray(self.data["labels"], dtype=int)
        SAVs = self.data["SAVs"]

        # ----- hold-out test split first -----
        all_idx = np.arange(self.nSAVs)
        train_idx, test_idx = train_test_split(
            all_idx,
            test_size=cfg.test_size,
            random_state=cfg.seed,
            stratify=y
        )
        x_te, y_te, sav_te  = X[test_idx],  y[test_idx],  SAVs[test_idx]

        fd_models = self.models
        # ----- CV on training set -----
        skf = StratifiedKFold(n_splits=cfg.val_splits, shuffle=True, random_state=cfg.seed)
        models = []
        history = {
            'fd': {'val': [], 'test': []}, 
            'tf': {'val': [], 'test': []}
        }

        for fold_idx, (inner_tr, inner_va) in enumerate(skf.split(train_idx, y[train_idx]), start=1):
            x_tr, y_tr, sav_tr = X[inner_tr], y[inner_tr], SAVs[inner_tr]
            x_va, y_va, sav_va = X[inner_va], y[inner_va], SAVs[inner_va]
            
            model_dir = os.path.join(job_dir, f'fold_{fold_idx}')
            os.makedirs(model_dir, exist_ok=True)

            # log the folds
            pos_tr  = int(np.sum(y_tr))
            neg_tr  = int(len(y_tr) - np.sum(y_tr))
            pos_va  = int(np.sum(y_va))
            neg_va  = int(len(y_va) - np.sum(y_va))
            pos_te  = int(np.sum(y_te))
            neg_te  = int(len(y_te) - np.sum(y_te))

            LOGGER.info(
                f"Fold {fold_idx} - Train: {pos_tr}pos + {neg_tr}neg, "
                f"Val: {pos_va}pos + {neg_va}neg, "
                f"Test: {pos_te}pos + {neg_te}neg"
            )
            LOGGER.info(f"Train: {sav_tr}")
            LOGGER.info(f"Val: {sav_va}")
            LOGGER.info(f"Test: {sav_te}")
            
            y_tr_1h = onehot_encoding(y_tr, 2)
            y_va_1h = onehot_encoding(y_va, 2)
            y_te_1h = onehot_encoding(y_te, 2)

            train_ds = np2ds(x_tr,   y_tr_1h,   shuffle=True,  batch_size=cfg.batch_size, seed=cfg.seed)
            val_ds   = np2ds(x_va,   y_va_1h,   shuffle=False, batch_size=cfg.batch_size, seed=cfg.seed)
            test_ds  = np2ds(x_te,   y_te_1h,   shuffle=False, batch_size=cfg.batch_size, seed=cfg.seed)

            # ----- build/train model -----
            # Load foundation model
            for model_idx, fd_model in enumerate(fd_models, start=1):
                fd_model_cp = tf.keras.models.clone_model(fd_model)
                fd_model_cp.set_weights(fd_model.get_weights())
                # Transfer learning 
                tf_model = train_model(
                    train_ds, 
                    val_ds, 
                    cfg=cfg,
                    folder=model_dir, 
                    filename=f'{model_idx}',
                    model_input=fd_model_cp,
                )
                tf_model.name = name
                models.append(tf_model)

                # ----- Evaluation -----
                fd_val_eval = fd_model.evaluate(val_ds, verbose=0)
                fd_test_eval = fd_model.evaluate(test_ds, verbose=0)
                tf_val_eval = tf_model.evaluate(val_ds, verbose=0)
                tf_test_eval = tf_model.evaluate(test_ds, verbose=0)

                metrics = ["loss", "accuracy", "auc", "precision", "recall", "f1"]
                left_title  = f"Foundation model (model {model_idx} fold {fold_idx})"
                right_title = "Transfer learning model"
                LOGGER.info(f"{left_title:<33} | {right_title}")
                LOGGER.info(f"{'val':>16}{'test':>11} {' ':>6}| {'val':>5}{'test':>11}")
                for idx, metric in enumerate(metrics):
                    left  = f"{fd_val_eval[idx]:>6.2f}{fd_test_eval[idx]:>10.2f}"
                    right = f"{tf_val_eval[idx]:>6.2f}{tf_test_eval[idx]:>10.2f}"
                    LOGGER.info(f"{metric:>9}: {left:<20}   | {right:<11}")

                history['fd']['val'].append(fd_val_eval)
                history['fd']['test'].append(fd_test_eval)
                history['tf']['val'].append(tf_val_eval)
                history['tf']['test'].append(tf_test_eval)
                
        self.history_avg(history)
        self.history_to_csv(history, filename=filename)
        LOGGER.report('train in %.1fs.', '_train')
        return history