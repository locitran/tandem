import os
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, addpath)

from tandem.src.main import tandem_dimple

query = 'P29033'
# td = tandem_dimple(
#     query=query, # List of SAVs to be analyzed
#     job_name='input_as_uniprot_test', # Define where the job will be saved
#     # custom_PDB='2ZW3', # Path to the custom PDB file (if any)
#     custom_PDB=os.path.join(addpath, 'tandem/data/GJB2/structures/8qa2_opm_25Apr03.pdb'), # Path to the custom PDB file (if any)
#     refresh=False, # Set to True to refresh the data
#     )

td = tandem_dimple(
    query=query, # List of SAVs to be analyzed
    job_name='P29033', # Define where the job will be saved
    custom_PDB='2ZW3', # Path to the custom PDB file (if any)
    # custom_PDB=os.path.join(addpath, 'tandem/data/GJB2/structures/8qa2_opm_25Apr03.pdb'), # Path to the custom PDB file (if any)
    refresh=True, # Set to True to refresh the data
    )