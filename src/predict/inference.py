import os 
import numpy as np
import tensorflow as tf
import pandas as pd

from prody import LOGGER

from ..train.modules import Preprocessing
from ..features import TANDEM_FEATS
from ..utils.settings import TANDEM_R20000, TANDEM_v1dot1


featSet = TANDEM_FEATS['v1.1']

class ModelInference:
    """This class is used to load the models and make predictions on the test data.
    Given feature matrix, it will return the predictions of the models.
    1. Load the models
    2. Load the training data (R20000 set)
    3. Preprocess the test data
    4. Get the predictions from the models
    5. Return the predictions
    """
    def __init__(self, folder=None, r20000=None, featSet=None):
        

        if folder is None:
            folder = TANDEM_v1dot1
        if r20000 is None:
            r20000 = TANDEM_R20000
        if featSet is None:
            featSet = TANDEM_FEATS['v1.1']

        self.models = self.importModels(folder)
        self.r20000 = r20000
        self.featSet = featSet
        fm = self.importR20000(r20000, featSet)
        self.p = Preprocessing(fm)

    # import models
    def importModels(self, folder):
        """Import models from the given folder.
        Args:
            folder (str): Folder containing the models.
        Returns:
            models (list): List of models.
        """
        assert os.path.exists(folder), f"Folder {folder} does not exist."
        models = []
        # for model in os.listdir(folder):
        #     if model.endswith('.h5'):
        #         models.append(os.path.join(folder, model))
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.h5'):
                    models.append(os.path.join(root, file))

        assert len(models) > 0, f"No models found in {folder}."
        LOGGER.info(f"Found {len(models)} models in {folder}.")
        LOGGER.info(f"Loading models from {folder}.")
        models = [tf.keras.models.load_model(model) for model in models]
        return models

    def importR20000(self, r20000, featSet):
        """Import the R20000 dataset.
        Args:
            r20000 (str): Path to the R20000 dataset.
        Returns:
            fm (np.ndarray): Feature matrix of the R20000 dataset. (n_samples, n_features)
        """
        assert os.path.exists(r20000), f"File {r20000} does not exist."
        df = pd.read_csv(r20000)
        fm = df[featSet].values
        return fm

    def calcPredictions(self, test_fm):
        """Make predictions on the test data.
        Args:
            test_fm (np.ndarray): Feature matrix of the test data. (n_samples, n_features)
        Returns:
            preds (np.ndarray): Predictions of the models. (n_samples, n_models)
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        LOGGER.info("Using CPU for inference. Set CUDA_VISIBLE_DEVICES to 0 to use GPU.")

        # Preprocess the test_fm
        test_fm = self.p.fill_na_mean(test_fm)
        test_fm = self.p.normalize(test_fm)
        
        # Get the predictions from the models
        preds = np.zeros((len(test_fm), len(self.models)))
        probs = np.zeros((len(test_fm), len(self.models)))
        for i, model in enumerate(self.models):
            prob = model.predict(test_fm)
            probs[:, i] = prob[:, 1]
            preds[:, i] = np.where(prob[:, 1] > 0.5, 1, 0)

        # Ratios are defined as the dominant prediction of the models
        # Get the ratio of the predictions
        votes = np.zeros((len(test_fm), 1))
        decisions = np.zeros((len(test_fm), 1))
        final_probs = np.zeros((len(test_fm), 1))
        for i in range(len(test_fm)):
            if np.sum(probs[i, :] > 0.5) > np.sum(probs[i, :] <= 0.5):
                decisions[i] = 1
                votes[i] = np.sum(probs[i, :] > 0.5) / len(probs[i, :])
                final_probs[i] = np.mean(probs[i, :][probs[i, :] > 0.5])
            else:
                decisions[i] = 0
                votes[i] = np.sum(probs[i, :] <= 0.5) / len(probs[i, :])
                final_probs[i] = np.mean(probs[i, :][probs[i, :] <= 0.5])

        self.probs = probs
        self.preds = preds
        self.votes = votes
        self.decisions = decisions
        self.final_probs = final_probs

    def get_prediction(self, test_fm):
        # Preprocess the test_fm
        self.preds = np.zeros((len(test_fm), len(self.models)))
        self.probs = np.zeros((len(test_fm), len(self.models)))
        for i, model in enumerate(self.models):
            prob = model.predict(test_fm)
            self.probs[:, i] = prob[:, 1]
            self.preds[:, i] = np.where(prob[:, 1] > 0.5, 1, 0)
        return self.preds
    
    def __call__(self, test_fm):
        test_fm = self.p.fill_na_mean(test_fm)
        test_fm = self.p.normalize(test_fm)
        return self.get_prediction(test_fm, self.models)
    
    def calcAccuracy(self, test_fm, test_labels):
        """Calculate the accuracy of the model on the test data.
        Args:
            test_fm (np.ndarray): Feature matrix of the test data. (n_samples, n_features)
            test_labels (np.ndarray): Labels of the test data. (n_samples, 1)
        Returns:
            accuracy (float): Accuracy of the model on the test data.
        """
        preds = self(test_fm) # (n_samples, n_models)
        # Calculate the accuracy of the model on the test data
        # Compare the predictions with the test labels
        accuracy = np.array(
            [np.sum(preds[:, i] == test_labels) / len(test_labels) for i in range(preds.shape[1])]
        )
        accuracy_mean = np.mean(accuracy)
        accuracy_sem = np.std(accuracy) / np.sqrt(len(accuracy))
        return accuracy, accuracy_mean, accuracy_sem
    
