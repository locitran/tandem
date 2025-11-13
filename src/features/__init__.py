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

rhapsody_feat = {
    'ANM_effectiveness_chain': r'Effectiveness$^*$',
    'ANM_sensitivity_chain': r'Sensitivity$^*$',
    'ANM_stiffness_chain': r'Stiffness$^*$',
    'wtPSIC': r'wtPSIC$^*$', 'deltaPSIC': r'$\Delta$PSIC$^*$', 
    'BLOSUM': r'BLOSUM$^*$', 'entropy': r'Entropy$^*$', 'ranked_MI': r'Ranked MI$^*$',
}

dynamics_feat = {
    'GNM_Ventropy_full': r'Entropy$_v$', 'GNM_rmsf_overall_full': r'RMSF$_{all}$', 
    'GNM_Eigval1_full': r'$\lambda_1$', 'GNM_Eigval2_full': r'$\lambda_2$',
    'GNM_Eigval5_1_full': r'$\lambda_{5-1}$', 'GNM_SEall_full': r'SE$_{all}$',
    'GNM_SE20_full': r'SE$_{20}$', 'GNM_V1_full': r'$‖V_{1,i}‖$',
    'GNM_rankV1_full': r'rank ($‖V_{1,i}‖$)', 'GNM_V2_full': r'$‖V_{2,i}‖$',
    'GNM_rankV2_full': r'rank ($‖V_{2,i}‖$)', 'GNM_co_rank_full': r'rank ($‖C_{i,i}‖$)',
    'GNM_displacement_full': r'$‖C_{i,i}‖$', 'GNM_MC1_full': r'MC${_1}$',
    'GNM_MC2_full': r'MC${_2}$', 'ANM_effectiveness_chain': r'Effectiveness$^*$',
    'ANM_sensitivity_chain': r'Sensitivity$^*$',
    'ANM_stiffness_chain': r'Stiffness$^*$'
    # 'wtBJCE': r'wtBJCE', 'deltaBJCE': r'$\Delta$BJCE'
}
structure_feat = {
    'chain_length': r'Protein Size', 
    'Rg': r'R$_g$', 'DELTA_Rg': r'$\Delta$R$_g$',
    'AG1': r'AG$_1$', 'AG3': r'AG$_3$', 'AG5': r'AG$_5$', 
    'ACR': r'ACR', 'DELTA_ACR': r'$\Delta$ACR',
    'SF1': r'SF$_1$', 'SF2': r'SF$_2$', 'SF3': r'SF$_3$',
    'loop_percent': r'%Loop', 'helix_percent': r'%Helix', 'sheet_percent': r'%Sheet',
    'Lside': r'L$_{side}$', 'deltaLside': r'$\Delta$L$_{side}$', # not availabel yet
    'IDRs': r'Disorderliness', 'dssp': r'DSSP', 'Dcom': r'D$_{com}$', 
    'SASA': r'SA', 'DELTA_SASA': r'$\Delta$SA', 
    'Hbond': r'N$_{H-bond}$', 'DELTA_Hbond': r'$\Delta$N$_{H-bond}$',
    'SSbond': r'N$_{SS-bond}$', 'DELTA_DSS': r'$\Delta$N$_{SS-bond}$',
}
sequence_feat = {
    'wtPSIC': r'wtPSIC$^*$', 'deltaPSIC': r'$\Delta$PSIC$^*$', 
    'BLOSUM': r'BLOSUM$^*$', 'entropy': r'Entropy$^*$', 'ranked_MI': r'Ranked MI$^*$',
    'consurf': r'ConSurf', 'ACNR': r'ACNR',
    'phobic_percent': r'%Hydrophobic', 'delta_phobic_percent': r'$\Delta$%Hydrophobic',
    # 'philic_percent': r'%Hydrophilic', 'delta_philic_percent': r'$\Delta$%Hydrophilic',
    'charge': r'Charge', 'deltaCharge': r'$\Delta$Charge', 
    'polarity': r'Polarity', 'deltaPolarity': r'$\Delta$Polarity', 
    'charge_pH7': r'Charge$_{pH7}$', 'DELTA_charge_pH7': r'$\Delta$Charge$_{pH7}$',
}
all_feat = {**dynamics_feat, **structure_feat, **sequence_feat}