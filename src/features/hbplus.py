import numpy as np
import subprocess, os

from prody import parsePDB, AtomGroup, writePDB
from prody.utilities import which

from ..utils.logger import LOGGER
from ..utils.settings import ROOT_DIR
from ..download import fetchPDB

__all__ = ['execHBplus', 'parseHBplus', 'calcHbond']

HBPLUSEXE = ROOT_DIR + '/src/features/bin/hbplus'
def execHBplus(pdb, cutoff=3.9, folder='.'):
    hbplus = which('hbplus')
    if hbplus is None:
        hbplus = HBPLUSEXE
    if not os.path.isfile(pdb):
        pdb = fetchPDB(pdb, compressed=False, folder=folder)
    if pdb is None:
        raise ValueError('pdb is not a valid PDB identifier or filename')
    filename = os.path.basename(pdb).split('.')[0]
    hb2 = os.path.join(folder, f'{filename}.hb2')
    debug = os.path.join(folder, 'hbdebug.dat')
    command = [hbplus, '-d', str(cutoff), '-A', '120', '0', '0', '-e', 'HOH', '" O "', '0', pdb]
    result = subprocess.run(command, cwd=folder, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    if result.returncode != 0:
        raise ValueError(f"Error in execHBplus: {result.stderr}")
    try:
        os.remove(debug)
    except Exception:
        pass
    return os.path.abspath(hb2)

def parseHBplus(hb2, ag, removefile=True, skip=1):
    """Parse HBplus output file and return hydrogen bond information for each residue.
    """
    if os.path.isfile(hb2):
        with open(hb2, 'r') as f:
            lines = f.readlines()
        if removefile:
            os.remove(hb2)
    else:
        lines = hb2.splitlines(keepends=True)
    lines = lines[8:]
    if not isinstance(ag, AtomGroup):
        raise ValueError('ag must be an AtomGroup instance')

    n_atoms = ag.numAtoms()
    HBOND = np.zeros(n_atoms, int)
    for line in lines:
        donor_chid = line[0].strip()
        donor_resid = int(line[1:5].strip())
        donor_icode = line[5].strip('-')

        acceptor_chid = line[14].strip()
        acceptor_resid = int(line[15:19].strip())
        acceptor_icode = line[19].strip('-')
        if donor_chid == acceptor_chid and abs(donor_resid - acceptor_resid) > skip:
            donor = ag[(donor_chid, donor_resid, donor_icode)]
            acceptor = ag[(acceptor_chid, acceptor_resid, acceptor_icode)]
            if donor is None or acceptor is None:
                continue
            donor_indices = donor.getIndices()
            acceptor_indices = acceptor.getIndices()
            HBOND[donor_indices] += 1
            HBOND[acceptor_indices] += 1
    ag.setData('hbond', HBOND)
    return ag

def calcHbond(pdbpath, chain_list="all", cutoff=3.9, 
              removefile=True, skip=1):
    folder = os.path.dirname(pdbpath)
    filename = os.path.basename(pdbpath).split('.')[0]
    if chain_list != "all":
        selchain = ""
        for chain in chain_list:
            selchain += f'chain `{chain}` or '
        selchain = selchain[:-4]

        pdb = parsePDB(pdbpath)
        pdb = pdb.select(selchain).copy()
        pdbpath = writePDB(os.path.join(folder, f'{filename}_{"".join(chain_list)}'), pdb.protein)
        # Add a HEADER line at the beginning of the file
        header_line = "HEADER    Generated PDB file\n"
        with open(pdbpath, "r") as f:
            pdb_content = f.readlines()
        with open(pdbpath, "w") as f:
            f.write(header_line)
            f.writelines(pdb_content[1:])
    else:
        pdb = parsePDB(pdbpath)
        pdb = pdb.protein

    LOGGER.timeit('_calcHBplusfeatures')
    hb2 = execHBplus(pdbpath, cutoff=cutoff)
    ag = parseHBplus(hb2, pdb, removefile=removefile, skip=skip)
    if chain_list != "all" and removefile:
        os.remove(pdbpath)
    LOGGER.report('HBplus features calculated in %.2fs.', label='_calcHBplusfeatures')
    return ag