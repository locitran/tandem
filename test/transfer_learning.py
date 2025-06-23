import os
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, addpath)

from tandem.src.train.train import reproduce_transfer_learning_model
import pandas as pd 
from tandem.src.features import TANDEM_FEATS

# Define feature set
feat_names = TANDEM_FEATS['v1.1']
# Features and labels
feat_path = os.path.join(addpath, 'tandem/data/GJB2/final_features.csv')
df = pd.read_csv(feat_path)
df = df[~df['labels'].isna()]
features = df[feat_names].values
labels = df['labels'].values
# Select foundation model
foundation_model = os.path.join(
    addpath, 'tandem/models/different_number_of_layers/20250423-1234-tandem/n_hidden-5/model_fold_1.h5')
# Run transfer learning
reproduce_transfer_learning_model(
    features, 
    labels, 
    name='reproduce_transfer_learning_model', 
    model_input=foundation_model, 
    seed=73, 
    patience = 50,
    n_epochs=300
)
