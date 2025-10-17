from pathlib import Path
import os 

__all__ = ['one2three', 'three2one', 'standard_aa', 'ROOT_DIR', 'RAW_PDB_DIR',
        'FIX_PDB_DIR', 'TMP_DIR', 'MATLAB_DIR', 'dynamics_feat', 
        'structure_feat', 'seq_feat', 'cols']

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
seq_feat = {
    'wtPSIC': r'wtPSIC$^*$', 'deltaPSIC': r'$\Delta$PSIC$^*$', 
    'BLOSUM': r'BLOSUM$^*$', 'entropy': r'Entropy$^*$', 'ranked_MI': r'Ranked MI$^*$',
    'consurf': r'ConSurf', 'ACNR': r'ACNR',
    'phobic_percent': r'%Hydrophobic', 'delta_phobic_percent': r'$\Delta$%Hydrophobic',
    'philic_percent': r'%Hydrophilic', 'delta_philic_percent': r'$\Delta$%Hydrophilic',
    'charge': r'Charge', 'deltaCharge': r'$\Delta$Charge', 
    'polarity': r'Polarity', 'deltaPolarity': r'$\Delta$Polarity', 
    'charge_pH7': r'Charge$_{pH7}$', 'DELTA_charge_pH7': r'$\Delta$Charge$_{pH7}$',
}
cols = {**dynamics_feat, **structure_feat, **seq_feat}

TANDEM_v1dot1 = os.path.join(ROOT_DIR, 'models', 'TANDEM')
TANDEM_v1dot1_GJB2 = os.path.join(ROOT_DIR, 'models', 'TANDEM_GJB2')
TANDEM_v1dot1_RYR1 = os.path.join(ROOT_DIR, 'models', 'TANDEM_RYR1')