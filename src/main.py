import os 
import datetime
import numpy as np
from .core import Tandem
from .utils.settings import ROOT_DIR
from .utils.logger import LOGGER
from .utils.settings import TANDEM_v1dot1

__all__ = ['run']

def run(
    query,
    labels=None,
    custom_PDB=None,
    pretrained_model_folder=TANDEM_v1dot1,
    job_name='tandem-dimple',
    features= None, 
    tf_name=None,
    config=None, 
    featSet=None,
    refresh=False,
    pkl_folder='data',
):
    """
    query: 
        1. Single amino acid variant(s) <UniProtID> <resid> <wt> <mt>
        2. <UniprotID> <resid>
        3. <UniProtID>
    labels: 
        1: pathogenic; 0: benign
    custom_PDB: 
        1. uploaded coordinate file
        2. AlphaFold DB ID
        3. PDB ID
    pretrained_model_folder: 
        1. TANDEM_v1dot1, default TANDEM foundation models
        2. TANDEM_v1dot1_GJB2, TANDEM-DIMPLE for GJB2
        3. TANDEM_v1dot1_RYR1, TANDEM-DIMPLE for GJB2
        4. New pre-trained models
    """

    job_directory = os.path.join(ROOT_DIR, 'jobs', job_name)
    os.makedirs(job_directory, exist_ok=True)
    
    ## LOGGER
    logfile = os.path.join(job_directory, 'log.txt')
    LOGGER.start(logfile)
    LOGGER.info(f"Job name: {job_name} started at {datetime.datetime.now()}")
    LOGGER.info(f"Job directory: {job_directory}")
    LOGGER.timeit("_runtime")

    ## Save feature pickles
    os.makedirs(pkl_folder, exist_ok=True)

    # Set up the Tandem object
    t = Tandem(
        query, 
        refresh=refresh,
        job_directory=job_directory, 
        folder=pkl_folder,
    )
    t.getSAVs(filename='SAVs.txt', folder=job_directory)
    t.setFeatSet(featSet)
    
    if isinstance(features, np.ndarray):
        t.setFeatureMatrix(features)
    else:
        t.setCustomPDB(custom_PDB)
        t.getUniprot2PDBmap(filename='Uniprot2PDB.txt', folder=job_directory)
        t.getFeatMatrix(withSAVs=True, filename='features.csv', folder=job_directory)    

    history = None
    if labels:
        t.setLabels(labels)
        t.setConfig(config)
        name = tf_name if tf_name else job_name
        history = t.train(name, filename="history.csv")
    else:
        t.getPredictions(models=pretrained_model_folder, folder=job_directory, filename='predictions.txt')

    # SHAP analysis
    # Plot global feature contribution
    # 1. TANDEM models
    # 2. TANDEM-DIMPLE models
    t.plotSHAP(folder=job_directory)
    
    for label in LOGGER._reports:
        LOGGER.info(f"  {label}: {LOGGER._reports[label]:.2f}s ({LOGGER._report_times[label]} time(s))")
    LOGGER.report('Run time elapsed in %.2fs.', "_runtime")
    LOGGER.close(logfile)
    return t, history
