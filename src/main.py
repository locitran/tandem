import os 
import datetime

from .core import Tandem
from .utils.settings import ROOT_DIR
from .utils.logger import LOGGER
from .features.PolyPhen2 import printSAVlist

# from prody import LOGGER

__all__ = ['tandem_dimple']

def tandem_dimple(
    query, 
    job_name='tandem-dimple', 
    models=None,
    r20000=None, 
    custom_PDB=None,
    featSet=None,
    refresh=False,
    **kwargs
):
    """Main function to calculate features for SAVs."""
    # Create a directory for the job
    job_directory = os.path.join(ROOT_DIR, 'jobs', job_name)
    os.makedirs(job_directory, exist_ok=True)
    
    ## LOGGE
    logfile = os.path.join(job_directory, 'log.txt')
    LOGGER.start(logfile)
    LOGGER.info(f"Job name: {job_name} started at {datetime.datetime.now()}")
    LOGGER.info(f"Job directory: {job_directory}")
    LOGGER.timeit("_runtime")

    ## Additional arguments
    kwargs['job_directory'] = job_directory
    if 'folder' not in kwargs:
        kwargs['folder'] = os.path.join(ROOT_DIR, 'data') 
    os.makedirs(kwargs['folder'], exist_ok=True)

    # Set up the Tandem object
    t = Tandem(query, refresh=refresh, **kwargs)

    # Save SAVs to a file
    printSAVlist(t.data['SAV_coords'], os.path.join(job_directory, 'SAVs.txt'))

    # Set custom PDB structure
    if custom_PDB:
        t.setCustomPDB(custom_PDB)

    if featSet:
        # Set up the feature set
        t.setFeatSet(featSet)
    else:
        # Set up the default feature set
        t.setFeatSet('v1.1')
    
    # Save the Uniprot2PDB map
    t.getUniprot2PDBmap(folder=job_directory, filename=job_name)
    # Calculate the feature matrix
    t.getFeatMatrix(withSAVs=True, folder=job_directory, filename=job_name)
    # Calculate predictions
    t.predictSAVs(
        models=models, 
        r20000=r20000, 
        model_names='TANDEM-DIMPE_v1',
        folder=job_directory,
        filename=job_name
    )
    
    for label in LOGGER._reports:
        LOGGER.info(f"  {label}: {LOGGER._reports[label]:.2f}s ({LOGGER._report_times[label]} time(s))")
        
    LOGGER.report('Run time elapsed in %.2fs.', "_runtime")
    LOGGER.close(logfile)
    return t