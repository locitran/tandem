import subprocess, os
import numpy as np

from prody.atomic import ATOMIC_FIELDS
from prody.atomic import AtomGroup
from prody.utilities import which
from prody import parsePDB, writePDB

from ..download import fetchPDB
from ..utils.settings import ROOT_DIR
from ..utils.logger import LOGGER


__all__ = ['execDSSP', 'parseDSSP', 'calcDSSP', 'calcSecondary']

DSSPEXE = ROOT_DIR + '/src/features/bin/mkdssp'

def execDSSP(pdb, filename=None, folder='.'):
    dssp = which('mkdssp')
    if dssp is None:
        dssp = which('dssp')
    if dssp is None:
        dssp = DSSPEXE
    if not os.path.isfile(pdb):
        pdb = fetchPDB(pdb, compressed=False)
    if pdb is None:
        raise ValueError('pdb is not a valid PDB identifier or filename')
    command = [dssp, pdb, '--output-format', 'dssp']
    result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise ValueError(f"Error in execDSSP: {result.stderr}")
    if filename is not None:
        out = os.path.join(folder, filename + '.dssp')
        with open(out, 'w') as f:
            f.write(result.stdout)
        return out
    else:
        return result.stdout

def parseDSSP(dssp, ag, parseall=False, removefile=False):
    """
    Parse DSSP output file and assign secondary structure information to atoms in an AtomGroup instance.
    This is a modified version of the original parseDSSP function from ProDy.
    
    Reference
    ---------
    https://swift.cmbi.umcn.nl/gv/dssp/
    """
    if os.path.isfile(dssp):
        with open(dssp, 'r') as f:
            lines = f.readlines()
        if removefile:
            os.remove(dssp)
    else:
        lines = dssp.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith('  #  RESIDUE'):
            break
    lines = lines[i+1:]
    if not isinstance(ag, AtomGroup):
        raise TypeError('ag argument must be an AtomGroup instance')
    
    n_atoms = ag.numAtoms()     
    NUMBER = np.zeros(n_atoms, int)
    SHEETLABEL = np.zeros(n_atoms, np.array(['a']).dtype.char + '1')
    ACC = np.zeros(n_atoms, float)
    KAPPA = np.zeros(n_atoms, float)
    ALPHA = np.zeros(n_atoms, float)
    PHI = np.zeros(n_atoms, float)
    PSI = np.zeros(n_atoms, float)
    
    if parseall:
        BP1 = np.zeros(n_atoms, int)
        BP2 = np.zeros(n_atoms, int)
        NH_O_1 = np.zeros(n_atoms, int)
        NH_O_1_nrg = np.zeros(n_atoms, float)
        O_HN_1 = np.zeros(n_atoms, int)
        O_HN_1_nrg = np.zeros(n_atoms, float)
        NH_O_2 = np.zeros(n_atoms, int)
        NH_O_2_nrg = np.zeros(n_atoms, float)
        O_HN_2 = np.zeros(n_atoms, int)
        O_HN_2_nrg = np.zeros(n_atoms, float)
        TCO = np.zeros(n_atoms, float)

    ag.setSecstrs(np.zeros(n_atoms, dtype=ATOMIC_FIELDS['secondary'].dtype))
    for line in lines:
        if line[13] == '!':
            continue
        res = ag[(line[11], int(line[5:10]), line[10].strip())]
        if res is None:
            continue
        indices = res.getIndices()
        res.setSecstrs(line[16].strip())
        NUMBER[indices] = int(line[:5])
        SHEETLABEL[indices] = line[33].strip()
        ACC[indices] = int(line[35:38])
        KAPPA[indices] = float(line[91:97])
        ALPHA[indices] = float(line[97:103])
        PHI[indices] = float(line[103:109])
        PSI[indices] = float(line[109:115])
        
        if parseall:
            BP1[indices] = int(line[25:29])
            BP2[indices] = int(line[29:33])
            NH_O_1[indices] = int(line[38:45])
            NH_O_1_nrg[indices] = float(line[46:50])
            O_HN_1[indices] = int(line[50:56])
            O_HN_1_nrg[indices] = float(line[57:61])
            NH_O_2[indices] = int(line[61:67])
            NH_O_2_nrg[indices] = float(line[68:72])
            O_HN_2[indices] = int(line[72:78])
            O_HN_2_nrg[indices] = float(line[79:83])
            TCO[indices] = float(line[85:91])

    ag.setData('dssp_resnum', NUMBER)
    ag.setData('dssp_sheet_label', SHEETLABEL)
    ag.setData('dssp_acc', ACC)
    ag.setData('dssp_kappa', KAPPA)
    ag.setData('dssp_alpha', ALPHA)
    ag.setData('dssp_phi', PHI)
    ag.setData('dssp_psi', PSI)

    if parseall:
        ag.setData('dssp_bp1', BP1)
        ag.setData('dssp_bp2', BP2)
        ag.setData('dssp_NH_O_1_index', NH_O_1)
        ag.setData('dssp_NH_O_1_energy', NH_O_1_nrg)
        ag.setData('dssp_O_NH_1_index', O_HN_1)
        ag.setData('dssp_O_NH_1_energy', O_HN_1_nrg)
        ag.setData('dssp_NH_O_2_index', NH_O_2)
        ag.setData('dssp_NH_O_2_energy', NH_O_2_nrg)
        ag.setData('dssp_O_NH_2_index', O_HN_2)
        ag.setData('dssp_O_NH_2_energy', O_HN_2_nrg)
        ag.setData('dssp_tco', TCO)
    return ag

def calcDSSP(pdb, parseall=False, removefile=False):
    dssp = execDSSP(pdb)
    ag = parsePDB(pdb)
    return parseDSSP(dssp, ag, parseall=parseall, removefile=removefile)

def calcSecondary(pdbpath, chain="all", removefile=True):
    """
    Get DSSP data from PDB file
    - Helices: 0.5
    H = a-helix
    I = π-helix
    G = 310-helix
    P = κ-helix (poly-proline II helix)
    - Sheets: 0
    B = residue in isolated β-bridge
    E = extended strand, participates in β ladder
    - Loops: 1
    T = hydrogen-bonded turn
    S = bend
    """
    folder = os.path.dirname(pdbpath)
    filename = os.path.basename(pdbpath).split('.')[0]
    if chain != "all":
        filename = f'{filename}_{chain}'
        pdb = parsePDB(pdbpath, compressed=False, folder=folder)
        # pdb = pdb.protein[chain].copy()
        pdb = pdb.protein.select(f'chain `{chain}`').copy()
        pdbpath = writePDB(os.path.join(folder, f'{filename}.pdb'), pdb)
        # Add a HEADER line at the beginning of the file
        header_line = "HEADER    Generated PDB file\n"
        with open(pdbpath, "r") as f:
            pdb_content = f.readlines()
        with open(pdbpath, "w") as f:
            f.write(header_line)
            f.writelines(pdb_content[1:])  # Write the rest of the file content
    else:
        pdb = parsePDB(pdbpath, compressed=False, folder=folder)

    LOGGER.timeit('_calcDSSPfeatures')
    dssp = execDSSP(pdbpath)
    ag = parseDSSP(dssp, pdb, removefile=removefile)
    if chain != "all" and removefile:
        os.remove(pdbpath)
    # Calculate secondary structure
    secondary = ag.ca.getSecstrs()
    secondary2score = {'H':0.5, 'I':0.5, 'G':0.5, 'P': 0.5, 
                       'T':1, 'S':1, '':1,
                       'B':0, 'E':0}
    secondary_score = np.array([secondary2score[s] for s in secondary])
    n_residues = secondary.shape[0]
    loop_percent = np.sum(secondary_score == 1) / n_residues * 100
    sheet_percent = np.sum(secondary_score == 0) / n_residues * 100
    helix_percent = np.sum(secondary_score == 0.5) / n_residues * 100
    LOGGER.report('DSSP features calculated in %.2fs.', label='_calcDSSPfeatures')
    return loop_percent, sheet_percent, helix_percent, secondary_score