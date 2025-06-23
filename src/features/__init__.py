# -*- coding: utf-8 -*-
"""This subpackage contains modules for computing features from multiple
sources, e.g. Uniprot sequences, PDB structures, Pfam domains and
EVmutation precomputed data.
"""

__author__ = "Luca Ponzoni"
__date__ = "December 2019"
__maintainer__ = "Luca Ponzoni"
__email__ = "lponzoni@pitt.edu"
__status__ = "Production"

__secondary_author__ = "Loci Tran"

__all__ = ['TANDEM_FEATS']

from . import Uniprot
from .Uniprot import *
__all__.extend(Uniprot.__all__)
__all__.append('Uniprot')

from . import PDB
from .PDB import *
__all__.extend(PDB.__all__)
__all__.append('PDB')

from . import PolyPhen2
from .PolyPhen2 import *
__all__.extend(PolyPhen2.__all__)
__all__.append('PolyPhen2')

from . import SEQ
from .SEQ import *
__all__.extend(SEQ.__all__)
__all__.append('SEQ')

# list of all available features in RHAPSODY
TANDEM_FEATS = {
    'PDB': set(PDB.PDB_FEATS),
    'SEQ': set(SEQ.SEQ_FEATS),
}
TANDEM_FEATS['all'] = set().union(*TANDEM_FEATS.values())
# Feature set used in my thesis August 2024
TANDEM_FEATS['v1.0'] = [
    "consurf", "wtPSIC", "deltaPSIC", "entropy", "ACNR", "SASA", "BLOSUM", "ANM_stiffness_chain",
    "loop_percent", "AG1", "GNM_V2_full", "GNM_co_rank_full", "AG3", "AG5", "Dcom", "GNM_V1_full",
    "GNM_rankV2_full", "GNM_Eigval1_full", "ranked_MI", "DELTA_Hbond", "phobic_percent", "GNM_Eigval2_full",
    "sheet_percent", "Rg", "deltaPolarity", "Lside", "helix_percent", "deltaLside", "ANM_effectiveness_chain",
    "GNM_rankV1_full", "GNM_rmsf_overall_full", "deltaCharge", "delta_phobic_percent"]
# Feature set used from May 2025 
TANDEM_FEATS['v1.1'] = [
    # DYN 9 features
    'GNM_co_rank_full', 'ANM_stiffness_chain', 'GNM_V2_full', 'GNM_V1_full', 'GNM_Eigval1_full', 
    'GNM_rankV2_full', 'GNM_Eigval2_full', 'GNM_rankV1_full', 'ANM_effectiveness_chain', 
    # STR 15 features
    'SASA', 'loop_percent', 'AG1', 'Dcom', 'AG5', 'AG3', 'SSbond', 'Hbond', 'DELTA_Hbond', 
    'sheet_percent', 'helix_percent', 'Rg', 'IDRs', 'Lside', 'deltaLside', 
    # SEQ 9 features
    'entropy', 'wtPSIC', 'deltaPSIC', 'consurf', 'ACNR', 'BLOSUM', 'ranked_MI', 'deltaPolarity', 'deltaCharge'
]
