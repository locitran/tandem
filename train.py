# from src.train.optimization import test_numberOflayers_TANDEM, test_numberOflayers_RHAPSODY, test_ranking_method, simple_training
# from src.train.optimization import test_batch_size, test_different_numberOfneurons, visualization_optimization
from src.train.train import reproduce_foundation_model, reproduce_transfer_learning_model
# from src.train.direct_train import train_model
from src.utils.settings import TANDEM_GJB2, TANDEM_RYR1, TANDEM_v1dot1
if __name__ == "__main__":
#     # train_model(
    #     base_models="/mnt/nas_1/YangLab/loci/tandem/logs/Optimization_Tandem_NumberOfLayers/20250627-1012/n_hidden-5",
    #     TANDEM_testSet=TANDEM_RYR1,
    #     name="RYR1",
    #     seed=100,
    # )
    # reproduce_foundation_model()
    # reproduce_transfer_learning_model(
    #     base_models="/mnt/nas_1/YangLab/loci/tandem/models/tandem",
    #     TANDEM_testSet=TANDEM_RYR1,
    #     name="RYR1",
    #     seed=0,
    # )
    
    reproduce_transfer_learning_model(
        base_models=TANDEM_v1dot1,
        TANDEM_testSet=TANDEM_GJB2,
        name="GJB2",
        seed=73,
    )

    # simple_training(
        # seed=17
    # )
    
    # test_numberOflayers_TANDEM(
    #     seed=17
    # )
    
# from src.train.train import reproduce_transfer_learning_model, reproduce_foundation_model
# import pandas as pd 
# from src.features import TANDEM_FEATS
# feat_names = TANDEM_FEATS['v1.1']
# feat_path = '/mnt/nas_1/YangLab/loci/tandem/data/GJB2/final_features.csv'
# df = pd.read_csv(feat_path)
# df = df[~df['labels'].isna()]
# features = df[feat_names].values
# labels = df['labels'].values

# reproduce_transfer_learning_model(
#     features, 
#     labels, 
#     name='reproduce_transfer_learned_model', 
#     model_input=None, 
#     seed=73, 
#     patience = 50
# )

# reproduce_foundation_model(name='weight_initialization')

"""
write new function(s) / module. 

Take three inputs 



goal: execute the file automatically. 

input: SAVs and labels
hyper-parameters (default) advance options



"""

