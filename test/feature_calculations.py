import os
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, addpath) # /home/newloci

from tandem.src.main import tandem_dimple
from tandem.src.core import calcFeatures

query = ['O14508 52 S N', 'P29033 217 Y D']

td = calcFeatures(
    query=query, # List of SAVs to be analyzed
    job_name='feature_calculations', # Define where the job will be saved
    custom_PDB=None, # Path to the custom PDB file (if any)
    refresh=False, # Set to True to refresh the calculation
    )   
