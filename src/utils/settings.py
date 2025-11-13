from pathlib import Path
import os 

__all__ = ['one2three', 'three2one', 'standard_aa', 'ROOT_DIR', 'RAW_PDB_DIR',
        'FIX_PDB_DIR', 'TMP_DIR', 'MATLAB_DIR']

aa_list = 'ACDEFGHIKLMNPQRSTVWY'
one2three = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY',
    'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
    'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER', 'T': 'THR', 'V': 'VAL',
    'W': 'TRP', 'Y': 'TYR'
}
# one2three = {
#     'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN',
#     'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE',
#     'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
# }
three2one = {v: k for k, v in one2three.items()}
standard_aa = list(one2three.values())
ROOT_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = str(ROOT_DIR)
RAW_PDB_DIR = ROOT_DIR + '/pdbfile/raw'
FIX_PDB_DIR = ROOT_DIR + '/pdbfile/fix'
TMP_DIR = ROOT_DIR + '/src/features/tmp'
MATLAB_DIR = ROOT_DIR + '/src/features/matlab'

CLUSTER = ROOT_DIR + '/data/R20000/c30_clstr_May13.csv'
CLUSTER = ROOT_DIR + '/data/c30_clstr_May13_full_rhd.csv'

FEAT_STATS = ROOT_DIR + '/data/R20000/stats/features_stats.csv'
TANDEM_R20000 = ROOT_DIR + '/data/R20000/final_features.csv'
TANDEM_GJB2 = ROOT_DIR + '/data/GJB2/final_features.csv'
TANDEM_RYR1 = ROOT_DIR + '/data/RYR1/RYR1-features.csv'
# TANDEM_RYR1 = ROOT_DIR + '/data/RYR1/final_features.csv'
TANDEM_PKD1 = ROOT_DIR + '/data/PKD1/final_features_PKD1.csv'

RHAPSODY_R20000 = ROOT_DIR + '/data/R20000/rhd_final_features.csv'
RHAPSODY_GJB2 = ROOT_DIR + '/data/GJB2/rhd_final_features.csv'
RHAPSODY_RYR1 = ROOT_DIR + '/data/RYR1/rhd_final_features.csv'
RHAPSODY_PKD1 = ROOT_DIR + '/data/PKD1/rhd_final_features_PKD1.csv'
RHAPSODY_FEATS = ['ANM_MSF-chain', 'ANM_MSF-reduced', 'ANM_MSF-sliced', 'ANM_effectiveness-chain', 'ANM_effectiveness-reduced', 'ANM_effectiveness-sliced', 'ANM_sensitivity-chain', 'ANM_sensitivity-reduced', 'ANM_sensitivity-sliced', 'BLOSUM', 'Delta_PSIC', 'Delta_SASA', 'EVmut-DeltaE_epist', 'EVmut-DeltaE_indep', 'EVmut-mut_aa_freq', 'EVmut-wt_aa_cons', 'GNM_MSF-chain', 'GNM_MSF-reduced', 'GNM_MSF-sliced', 'GNM_effectiveness-chain', 'GNM_effectiveness-reduced', 'GNM_effectiveness-sliced', 'GNM_sensitivity-chain', 'GNM_sensitivity-reduced', 'GNM_sensitivity-sliced', 'SASA', 'SASA_in_complex', 'entropy', 'ranked_MI', 'stiffness-chain', 'stiffness-reduced', 'stiffness-sliced', 'wt_PSIC']

TANDEM_v1dot1 = os.path.join(ROOT_DIR, 'models', 'TANDEM')
TANDEM_v1dot1_GJB2 = os.path.join(ROOT_DIR, 'models', 'TANDEM_GJB2')
TANDEM_v1dot1_RYR1 = os.path.join(ROOT_DIR, 'models', 'TANDEM_RYR1')