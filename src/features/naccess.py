import numpy as np
import subprocess, os

from prody import parsePDB, AtomGroup, writePDB
from prody.utilities import which

from ..utils.logger import LOGGER
from ..utils.settings import ROOT_DIR
from ..download import fetchPDB

__all__ = ['execNACCESS', 'parseNACCESS', 'calcAccessibility']

NACCESSEXE = ROOT_DIR + '/src/features/bin/naccess'
def execNACCESS(pdb, filename=None, folder='.'):
    naccess = which('naccess')
    if naccess is None:
        naccess = NACCESSEXE
    if not os.path.isfile(pdb):
        pdb = fetchPDB(pdb, compressed=False, folder=folder)
    if pdb is None:
        raise ValueError('pdb is not a valid PDB identifier or filename')
    filename = os.path.basename(pdb).split('.')[0]
    logf = os.path.join(folder, f'{filename}.log')
    asaf = os.path.join(folder, f'{filename}.asa')
    rsaf = os.path.join(folder, f'{filename}.rsa')
    command = [naccess, pdb]
    result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise ValueError(f"Error in execNACCESS: {result.stderr}")
    try:
        os.remove(logf)
        os.remove(asaf)
    except Exception:
        pass
    return rsaf

def parseNACCESS(naccess, ag, removefile=False):
    """Parse NACCESS output file and return ASA and RSA values for each residue.
    REM  File of summed (Sum) and % (per.) accessibilities for 
    REM RES _ NUM      All-atoms   Total-Side   Main-Chain    Non-polar    All polar
    REM                ABS   REL    ABS   REL    ABS   REL    ABS   REL    ABS   REL

    aa_abs: All-atoms absolute accessibility
    aa_rel: All-atoms relative accessibility
    ts_abs: Total-side absolute accessibility
    ts_rel: Total-side relative accessibility
    mc_abs: Main-chain absolute accessibility
    mc_rel: Main-chain relative accessibility
    np_abs: Non-polar absolute accessibility
    np_rel: Non-polar relative accessibility
    ap_abs: All-polar absolute accessibility
    ap_rel: All-polar relative accessibility
    """
    if os.path.isfile(naccess):
        with open(naccess, 'r') as f:
            lines = f.readlines()
    else:
        lines = naccess.splitlines(keepends=True)
    lines = [line for line in lines if line.startswith('RES')]
    if removefile:
        try:
            os.remove(naccess)
        except Exception:
            pass
    if not isinstance(ag, AtomGroup):
        raise ValueError('ag must be an AtomGroup instance')
    
    n_atoms = ag.numAtoms()
    RESNUM = np.zeros(n_atoms, int)
    names = ['AA_ABS', 'AA_REL', 'TS_ABS', 'TS_REL', 'MC_ABS', 'MC_REL', 'NP_ABS', 'NP_REL', 'AP_ABS', 'AP_REL']
    _dict = {name: np.zeros(n_atoms, float) for name in names}

    for line in lines:
        try:
            res = ag[(line[7:9].strip(), int(line[9:13]), line[13].strip())]
            if res is None:
                continue
            indices = res.getIndices()
            RESNUM[indices] = int(line[9:13])
            rsa = line[14:].strip().split()
            for name, val in zip(names, rsa):
                _dict[name][indices] = float(val)
        except Exception as e:
            LOGGER.warn(f"{line[7:9].strip()}, {int(line[9:13])}, {line[13].strip()}")
            LOGGER.warn(f"{ag[(line[7:9].strip(), int(line[9:13]), line[13].strip())]}")
            LOGGER.warn(e)
            continue
    
    ag.setData('rsa_resnum', RESNUM)
    for name, val in _dict.items():
        ag.setData(f'naccess_{name.lower()}', val)
    return ag

def calcAccessibility(pdbpath, chain="all", removefile=True):
    folder = os.path.dirname(pdbpath)
    filename = os.path.basename(pdbpath).split('.')[0]
    if chain != "all":
        filename = f'{filename}_{chain}'
        pdb = parsePDB(pdbpath)
        pdb = pdb.protein[chain].copy()
        pdbpath = writePDB(os.path.join(folder, f'{filename}.pdb'), pdb)
        # Add a HEADER line at the beginning of the file
        header_line = "HEADER    Generated PDB file\n"
        with open(pdbpath, "r") as f:
            pdb_content = f.readlines()
        with open(pdbpath, "w") as f:
            f.write(header_line)
            f.writelines(pdb_content[1:])  # Write the rest of the file content
    else:
        pdb = parsePDB(pdbpath)

    LOGGER.timeit('_calcNACCESSfeatures')
    naccess = execNACCESS(pdbpath)
    ag = parseNACCESS(naccess, pdb, removefile=removefile)
    if chain != "all" and removefile:
        os.remove(pdbpath)
    # aa_rel = ag.getData('naccess_aa_rel')
    LOGGER.report(f'NACCESS features (chain {chain}) calculated in %.2fs.', label='_calcNACCESSfeatures')
    return ag