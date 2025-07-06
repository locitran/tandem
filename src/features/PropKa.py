from propka import run
import pandas as pd
import numpy as np
from prody import parsePDB, writePDB
import os

from ..utils.logger import LOGGER
__all__ = ['parsePropKa', 'calcChargepH7']

def parsePropKa(mol, ag, pHCondition=7.0):
    n_atoms = ag.numAtoms()
    CHARGE_PH7 = np.zeros(n_atoms)

    groups = mol.conformations['AVR'].groups
    for group in groups:
        # res = ag[(line[7:9].strip(), int(line[9:13]), line[13].strip())]
        res = ag[
            (group.atom.chain_id, group.atom.res_num, group.atom.icode.strip())
        ]
        if res is None:
            continue
        indices = res.getIndices()
        if (group.charge > 0) == (group.pka_value > pHCondition):
            CHARGE_PH7[indices] = group.charge
    ag.setData('charge_pH7', CHARGE_PH7)
    return ag

def calcChargepH7(pdbpath, chain="all", pHCondition=7.0, write_pka=False, removefile=True):
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
    LOGGER.timeit('_calcPropKa')
    try: 
        mol = run.single(pdbpath, write_pka=write_pka)
    except:
        raise ValueError("Error in running propka for %s" % pdbpath)
    ag = parsePropKa(mol, pdb, pHCondition=pHCondition)
    if chain != "all" and removefile:
        try:
            os.remove(pdbpath)
        except:
            pass
    LOGGER.report(f'pKa features (chain {chain}) calculated in %.2fs.', label='_calcPropKa')
    return ag