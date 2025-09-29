import os 
import datetime
import numpy as np
from .core import Tandem
from .utils.settings import ROOT_DIR
from .utils.logger import LOGGER

__all__ = ['tandem_dimple']

def tandem_dimple(
    query,
    job_name='tandem-dimple',
    features= None, 
    tf_name=None,
    labels=None,
    config=None, 
    custom_PDB=None,
    featSet=None,
    refresh=False,
    pkl_folder='data',
):
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
        t.getUniprot2PDBmap(folder=job_directory, filename='Uniprot2PDB.txt')
        t.setCustomPDB(custom_PDB)
        t.getFeatMatrix(withSAVs=True, folder=job_directory, filename='features.csv')    
        
    if labels:
        t.setLabels(labels)
        t.setConfig(config)
        name = tf_name if tf_name else job_name
        t.train(name, filename="history.csv")
    else:
        t.getPredictions(folder=job_directory, filename='predictions.txt')

    for label in LOGGER._reports:
        LOGGER.info(f"  {label}: {LOGGER._reports[label]:.2f}s ({LOGGER._report_times[label]} time(s))")
    LOGGER.report('Run time elapsed in %.2fs.', "_runtime")
    LOGGER.close(logfile)
    return t
