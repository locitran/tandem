import numpy as np
import os
import traceback

from prody import Atomic, parsePDB, LOGGER
from prody.utilities import checkCoords
from prody.measure.contacts import findNeighbors
from prody import parsePDB, calcCenter
from prody import LOGGER, findNeighbors, parsePDB, writePDB, AtomGroup

from ..utils.settings import one2three

file_dir = os.path.dirname(os.path.abspath(__file__))

def calcLside(wt_aas, mut_aas):
    Lside_dict = {'G': 0, 'A': 1, 'V': 3, 'L': 4, 'I': 4, 'M': 4, 'F': 7, 'W': 10, 'P': 3, 'S': 2, 
              'T': 3, 'C': 2, 'Y': 8, 'N': 4, 'Q': 5, 'D': 4, 'E': 5, 'K': 5, 'R': 7, 'H': 6}
    
    assert len(wt_aas) == len(mut_aas), 'Length of wt_aas and mut_aas must be the same.'
    Lside = np.array([Lside_dict[mut] for mut in mut_aas])
    deltaLside = Lside - np.array([Lside_dict[wt] for wt in wt_aas])
    return Lside, deltaLside

def getBJCE(res1, res2, solvent=True, cutoff=6.4):
    """
    Retrieve the contact energy between two residues.
    
    Parameters:
    res1 (str): First residue name (e.g., "GLY").
    res2 (str): Second residue name (e.g., "ALA").
    solvent (bool): If True, includes solvent interaction (default=True).
    
    Returns:
    float: Contact energy value.

    Reference: Inter-residue potentials in globular proteins and the dominance of 
    highly specific hydrophilic interactions at close separation. Bahar & Jernigan, J. Mol. Biol. (1997)
    """
    eAB0_6dot4 = os.path.join(file_dir, 'eABo_6dot4.npy')
    eAB0_4 = os.path.join(file_dir, 'eABo_4.npy')

    eAB0_6dot4 = np.load(eAB0_6dot4, allow_pickle=True)
    eAB0_4 = np.load(eAB0_4, allow_pickle=True)

    # Define the residue names in order
    residue_names = [
        "GLY", "ALA", "VAL", "ILE", "LEU", "SER", "THR", "ASP", "ASN", "GLU",
        "GLN", "LYS", "ARG", "CYS", "MET", "PHE", "TYR", "TRP", "HIS", "PRO"
    ]

    aa_correction = {
        # Histidine (His)
        'HSD': 'HIS',   # NAMD, protonated at ND1 (HID in AMBER)
        'HSE': 'HIS',   # NAMD, protonated at NE2 (HIE in AMBER)
        'HSP': 'HIS',   # NAMD, doubly protonated (HIP in AMBER)
        'HID': 'HIS',   # AMBER name, protonated at ND1
        'HIE': 'HIS',   # AMBER name, protonated at NE2
        'HIP': 'HIS',   # AMBER name, doubly protonated
        'HISD': 'HIS',  # GROMACS: protonated at ND1
        'HISE': 'HIS',  # GROMACS: protonated at NE2
        'HISP': 'HIS',  # GROMACS: doubly protonated

        # Cysteine (Cys)
        'CYX': 'CYS',   # Cystine (disulfide bridge)
        'CYM': 'CYS',   # Deprotonated cysteine, anion

        # Aspartic acid (Asp)
        'ASH': 'ASP',   # Protonated Asp
        'ASPP': 'ASP',

        # Glutamic acid (Glu)
        'GLH': 'GLU',   # Protonated Glu
        'GLUP': 'GLU',  # Protonated Glu

        # Lysine (Lys)
        'LYN': 'LYS',   # Deprotonated lysine (neutral)

        # Arginine (Arg)
        'ARN': 'ARG',   # Deprotonated arginine (rare, GROMACS)

        # Tyrosine (Tyr)
        'TYM': 'TYR',   # Deprotonated tyrosine (GROMACS)

        # Serine (Ser)
        'SEP': 'SER',   # Phosphorylated serine (GROMACS/AMBER)

        # Threonine (Thr)
        'TPO': 'THR',   # Phosphorylated threonine (GROMACS/AMBER)

        # Tyrosine (Tyr)
        'PTR': 'TYR',   # Phosphorylated tyrosine (GROMACS/AMBER)

        # Non-standard names for aspartic and glutamic acids in low pH environments
        'ASH': 'ASP',   # Protonated Asp
        'GLH': 'GLU',   # Protonated Glu
    }

    # Ensure residues are valid
    res1 = aa_correction.get(res1, res1)
    res2 = aa_correction.get(res2, res2)
    if res1 not in residue_names or res2 not in residue_names:
        raise ValueError("Invalid residue name. Use standard residue codes.")
    # Get indices of the residues
    idx1 = residue_names.index(res1)
    idx2 = residue_names.index(res2)
    
    min_idx, max_idx = min(idx1, idx2), max(idx1, idx2)
    if solvent: # Upper triangle and diagonal
        if cutoff == 6.4:
            return eAB0_6dot4[min_idx, max_idx]
        else:
            return eAB0_4[min_idx, max_idx]
    else: # Lower triangle only
        if min_idx == max_idx:  # Same residue, no interaction
            return 0.0
        
        if cutoff == 6.4:
            return eAB0_6dot4[max_idx, min_idx] 
        else:
            return eAB0_4[max_idx, min_idx]
        
def calcBJCEnergy(pdb, chains, resids, wt_aas, mut_aas, icodes=None, solvent=True):
    """Calculate BJC energy for a PDB structure."""
    assert isinstance(pdb, (str, Atomic)), \
        'PDB must be a PDBID or an Atomic instance (e.g. AtomGroup).'
    assert len(chains) == len(resids) == len(wt_aas) == len(mut_aas), \
        'Length of chains, resids, and mut_aas must be the same.'
    if isinstance(pdb, str):
        pdb = parsePDB(pdb)
    if icodes is None:
        icodes = [''] * len(chains)
    pdb = pdb.ca

    _dtype = np.dtype([
        ('wtBJCE', 'f4'), ('mutBJCE', 'f4'), ('deltaBJCE', 'f4')
    ])
    features = np.full(len(chains), np.nan, dtype=_dtype)
    residues = [pdb[(chain, resid, icode)] for chain, resid, icode in zip(chains, resids, icodes)]
    for i, (res, wt_aa, mut_aa) in enumerate(zip(residues, wt_aas, mut_aas)):
        if res is None:
            LOGGER.warn(f"Residue {chains[i]} {resids[i]}{icodes[i]} not found in the structure.")
            continue
        resname = res.getResname()
        wt_resname = one2three[wt_aa]
        if wt_resname != resname:
            LOGGER.warn(f"Residue {chains[i]} {resids[i]}{icodes[i]} is not {wt_aa}.")
            continue
        mut_resname = one2three[mut_aa] 
        try:
            # Find the contact residues
            contact_4 = findNeighbors(res, 4, pdb)
            contact_6dot4 = findNeighbors(res, 6.4, pdb)
            # Filter out the contact residues
            contact_4 = [c[1] for c in contact_4 if c[2] > 0]
            contact_6dot4 = [c[1] for c in contact_6dot4 if c[2] > 0]
            contact_6dot4 = [c for c in contact_6dot4 if c not in contact_4]
            # Find the contact residue names
            contact_4 = [c.getResname() for c in contact_4]
            contact_6dot4 = [c.getResname() for c in contact_6dot4]
            # Calculate the BJC energy
            wtBJCE = 0
            mutBJCE = 0
            for contact_res in contact_4:
                wtBJCE += getBJCE(resname, contact_res, cutoff=4, solvent=solvent)
                mutBJCE += getBJCE(mut_resname, contact_res, cutoff=4, solvent=solvent)
            for contact_res in contact_6dot4:
                wtBJCE += getBJCE(resname, contact_res, cutoff=6.4, solvent=solvent)
                mutBJCE += getBJCE(mut_resname, contact_res, cutoff=6.4, solvent=solvent)
            deltaBJCE = mutBJCE - wtBJCE
            features[i] = wtBJCE, mutBJCE, deltaBJCE
        except Exception as e:
            raise ValueError(f"Error in calcBJCEnergy: {str(e)}")
    return features

def calcBJCEnergyFromFile(pdb, solvent=True, dump=False):
    """Calculate BJC energy from a PDB file."""
    assert isinstance(pdb, (str, Atomic)), \
        'PDB must be a PDBID or an Atomic instance (e.g. AtomGroup).'
    if isinstance(pdb, str):
        pdb = parsePDB(pdb)
    pdb = pdb.ca  # Use only CA atoms for contact calculation
    
    residues = list(pdb.protein.getHierView().iterResidues())
    dtype = np.dtype([
        ('residue', 'U50'), # Residue identifier (e.g., 'A_123A_GLY')
        ('c4', 'U50'),  # Contact residues within 4A
        ('c4_BJCE', 'U50'), # Contact residues within 4A with BJC energy
        ('c6dot4', 'U50'),  # Contact residues within 6.4A
        ('c6dot4_BJCE', 'U50'),  # Contact residues within 6.4A with BJC energy
        ('BJCE_per_residue', 'f4') # BJC energy per residue
    ])
    report = np.zeros(len(residues), dtype=dtype)

    for i, res in enumerate(residues):
        resname = res.getResname()
        chid = res.getChid()
        resnum = res.getResnum()
        icode = res.getIcode()
        key = f'{chid}_{resnum}{icode}_{resname}'
        try:
            contact_4 = findNeighbors(res, 4, pdb)
            # Filter out the contact residues
            contact_4 = [c[1] for c in contact_4 if c[2] > 0]
            # Find the contact residue names
            c4_resname = [c.getResname() for c in contact_4]
            c4_chid = [c.getChid() for c in contact_4]
            c4_resnum = [c.getResnum() for c in contact_4]
            c4_icode = [c.getIcode() for c in contact_4]
            contact_4_value = [f'{c4_chid[i]}_{c4_resnum[i]}{c4_icode[i]}_{c4_resname[i]}' for i in range(len(c4_resname))]
            
            contact_6dot4 = findNeighbors(res, 6.4, pdb)
            contact_6dot4 = [c[1] for c in contact_6dot4 if c[2] > 0]
            contact_6dot4 = [c for c in contact_6dot4 if c not in contact_4]
            c6dot4_resname = [c.getResname() for c in contact_6dot4]
            c6dot4_chid = [c.getChid() for c in contact_6dot4]
            c6dot4_resnum = [c.getResnum() for c in contact_6dot4]
            c6dot4_icode = [c.getIcode() for c in contact_6dot4]
            contact_6dot4_value = [f'{c6dot4_chid[i]}_{c6dot4_resnum[i]}{c6dot4_icode[i]}_{c6dot4_resname[i]}' for i in range(len(c6dot4_resname))]
            # Calculate the BJC energy
            c4_BJCE = []
            c6dot4_BJCE = []
            for contact_res in c4_resname:
                c4_BJCE.append(getBJCE(resname, contact_res, cutoff=4, solvent=solvent))
            for contact_res in c6dot4_resname:
                c6dot4_BJCE.append(getBJCE(resname, contact_res, cutoff=6.4, solvent=solvent))

            # LOGGER.info(f"Residue {key} has {len(contact_4_value)} contacts within 4A and {len(contact_6dot4_value)} contacts within 6.4A.")
            report[i]['residue'] = key
            report[i]['c4'] = ', '.join(contact_4_value)
            report[i]['c4_BJCE'] = ', '.join(map(str, c4_BJCE))
            report[i]['c6dot4'] = ', '.join(contact_6dot4_value)
            report[i]['c6dot4_BJCE'] = ', '.join(map(str, c6dot4_BJCE))
            report[i]['BJCE_per_residue'] = np.nansum(c4_BJCE) + np.nansum(c6dot4_BJCE)  # Total BJC energy per residue
        except Exception as e:
            # LOGGER.info(f"Error in calcBJCEnergyFromFile for residue {key}: {str(e)}")
            report[i]['residue'] = key
            report[i]['c4'] = ''
            report[i]['c4_BJCE'] = ''
            report[i]['c6dot4'] = ''
            report[i]['c6dot4_BJCE'] = ''
            report[i]['BJCE_per_residue'] = np.nan  # Set to NaN if error occurs
    if dump:
        # Dump the report to a file
        title = pdb.getTitle() 
        if solvent:
            title = f"{title}_solvent" 
        else: 
            title = f"{title}_non-solvent"
        output_file = f"{title}_BJC_energy_report.txt" if title else "BJC_energy_report.txt"
        with open(output_file, 'w') as f:
            # Write header
            f.write('Residue\tC4_Contacts\tC4_BJCE\tC6.4_Contacts\tC6.4_BJCE\tBJCE_per_residue\n')
            for row in report:
                f.write('\t'.join(map(str, row)) + '\n')
        LOGGER.info(f"BJC energy report saved to {output_file}")
    
    # Calculate BJCE of the whole structure
    total_BJCE = np.nansum(report['BJCE_per_residue'])
    LOGGER.info(f'Total BJC energy of the structure: {total_BJCE:.2f}')
    return total_BJCE

def calcAG(pdb, chain="all", cutoff=4.5, group_neighbor=[0, 1, 2], skip_neighbor=[1, 2, 3]):
    assert isinstance(pdb, (str, Atomic)), \
        'PDB must be a PDBID or an Atomic instance (e.g. AtomGroup).'
    if isinstance(pdb, str):
        pdb = parsePDB(pdb)
    try:
        if chain != 'all':
            resIndices = pdb.protein[chain].ca.copy().getResindices()
        else:
            resIndices = pdb.protein.ca.getResindices()
    except AttributeError as e:
        raise AttributeError(f'PDB object does not have protein attribute. {str(e)}')

    # Define group and skip neighbors
    group_neighbor=[0, 1, 2]
    skip_neighbor=[1, 2, 3]

    # Initialize AG
    _dtype = np.dtype([
        ('AG1', 'f'),
        ('AG3', 'f'),
        ('AG5', 'f')
    ])
    ag = np.full(len(resIndices), np.nan, dtype=_dtype)
    LOGGER.timeit('_ag')
    # Calculate AG
    residues = list(pdb.protein.getHierView().iterResidues())
    n_residues = len(residues)
    for i, resIndex in enumerate(resIndices):
        if (resIndex is None) or (resIndex < 0) or (resIndex >= n_residues):
            continue
        for name, group, skip in zip(['AG1', 'AG3', 'AG5'], group_neighbor, skip_neighbor):
            # Group residues
            resIndex_start = max(resIndex - group, 0)
            resIndex_end = min(resIndex + group + 1, n_residues)
            n_res_per_group = resIndex_end - resIndex_start
            group_residues = residues[resIndex_start:resIndex_end]
            group_coord = np.concatenate([res.getCoords() for res in group_residues])

            # Skip residues --> Unskip residues
            skip_resIndex_start = max(resIndex - group - skip, 0)
            skip_resIndex_end = min(resIndex + group + skip + 1, n_residues)
            unskip_residues = residues[:skip_resIndex_start] + residues[skip_resIndex_end:]
            unskip_coord = np.concatenate([res.getCoords() for res in unskip_residues])
            
            # Calculate contact matrix
            dist_m = np.sqrt(((group_coord[:, None, :] - unskip_coord[None, :, :]) ** 2).sum(axis=2)) # group_coord x unskip_coord
            contact_m = (dist_m < cutoff).astype(int)
            contact_m = contact_m.sum(0) # 1 x unskip_coord

            # Calculate group contact
            group_contact = (contact_m > 0).sum()
            group_contact = group_contact / n_res_per_group
            ag[i][name] = group_contact
    LOGGER.report(f'Atomic group contact (chain {chain}) calculation completed in %.2fs.', '_ag')
    return ag

def calcRGandDcom(pdb):
    assert isinstance(pdb, (str, Atomic)), \
        'PDB must be a PDBID or an Atomic instance (e.g. AtomGroup).'
    if isinstance(pdb, str):
        pdb = parsePDB(pdb)

    LOGGER.timeit('_calcRGandDcom')
    try:
        protein = pdb.protein
        residues = list(protein.getHierView().iterResidues())  # Use Hierarchical View to iterate residues
        res_centers = np.array([calcCenter(res) for res in residues])
        protein_center = np.mean(res_centers, axis=0)
        # Calculate radius of gyration
        rg = np.sqrt(np.sum(np.sum((res_centers - protein_center) ** 2, axis=1)) / res_centers.shape[0])
        # Calculate distance from the protein center to residue centers
        res_distances = np.sqrt(np.sum((res_centers - protein_center) ** 2, axis=1))
        dcom = (res_distances - np.mean(res_distances)) / np.std(res_distances)  # Standardize the distances
        LOGGER.report('Radius of gyration and distance from center of mass calculated in in %.2fs.', label='_calcRGandDcom')
        return rg, dcom
    except Exception as e:
        raise ValueError(f"Error in calcRGandDcom: {str(e)}")


def cleanNumbers(listContacts):
    """Provide short list with indices and value of distance."""
    
    shortList = [ [int(str(i[0]).split()[-1].strip(')')), 
                           int(str(i[1]).split()[-1].strip(')')), 
                           str(i[0]).split()[1], 
                           str(i[1]).split()[1], 
                           float(i[2])] for i in listContacts ]    
    return shortList

def removeDuplicates(list_of_interactions):
    """Remove duplicates from interactions."""
    ls=[]
    newList = []
    for no, i in enumerate(list_of_interactions):
       i = sorted(list(np.array(i).astype(str)))
       if i not in ls:
           ls.append(i)
           newList.append(list_of_interactions[no])
    return newList

def filterInteractions(list_of_interactions, atoms, **kwargs):
    """Return interactions based on *selection* and *selection2*."""
    
    if 'selection1' in kwargs:
        kwargs['selection'] = kwargs['selection1']

    if 'selection' in kwargs:
        selection = atoms.select(kwargs['selection'])
        if selection is None:
            LOGGER.warn('selection did not work, so no filtering is performed')
            return list_of_interactions

        ch1 = selection.getChids()
        x1 = selection.getResnames()
        y1 = selection.getResnums()
        listOfselection = np.unique(list(map(lambda x1, y1, ch1: (ch1, x1 + str(y1)),
                                             x1, y1, ch1)),
                                    axis=0)
        listOfselection = [list(i) for i in listOfselection] # needed for in check to work

        if 'selection2' in kwargs:
            selection2 = atoms.select(kwargs['selection2'])
            if selection2 is None:
                LOGGER.warn('selection2 did not work, so no filtering is performed')
                return list_of_interactions
            
            ch2 = selection2.getChids()
            x2 = selection2.getResnames()
            y2 = selection2.getResnums()
            listOfselection2 = np.unique(list(map(lambda x2, y2, ch2: (ch2, x2 + str(y2)),
                                                  x2, y2, ch2)),
                                         axis=0)
            listOfselection2 = [list(i) for i in listOfselection2] # needed for in check to work

            final = [i for i in list_of_interactions if (([i[2], i[0]] in listOfselection)
                                                         and ([i[5], i[3]] in listOfselection2)
                                                         or ([i[2], i[0]] in listOfselection2)
                                                         and ([i[5], i[3]] in listOfselection))]
        else:
            final = [i for i in list_of_interactions
                     if (([i[2], i[0]] in listOfselection)
                         or ([i[5], i[3]] in listOfselection))]

    elif 'selection2' in kwargs:
        LOGGER.warn('selection2 by itself is ignored')
        final = list_of_interactions
    else:
        final = list_of_interactions
    return final

def calcDisulfideBonds(atoms, distA=2.4, **kwargs):
    """Prediction of disulfide bonds.
    
    :arg atoms: an Atomic object from which residues are selected
    :type atoms: :class:`.Atomic`
    
    :arg distDB: non-zero value, maximal distance between atoms of cysteine residues.
        default is 3.
        distA works too
    :type distDB: int, float
    """

    try:
        coords = (atoms._getCoords() if hasattr(atoms, '_getCoords') else
                    atoms.getCoords())
    except AttributeError:
        try:
            checkCoords(coords)
        except TypeError:
            raise TypeError('coords must be an object '
                            'with `getCoords` method')
    
    from prody.measure import calcDihedral
    
    try:
        atoms_SG = atoms.select('protein and resname CYS and name SG')
        atoms_SG_res = list(set(zip(atoms_SG.getResnums(), atoms_SG.getChids())))
    
        LOGGER.info('Calculating disulfide bonds.')
        DisulfideBonds_list = []
        for i in atoms_SG_res:
            selstr = f'(same residue as protein within {distA} of (resid {i[0]} and chain {i[1]} and name SG)) ' + \
                        f'and (resname CYS and name SG)'
            CYS_pairs = atoms.select(selstr)
            if CYS_pairs.numAtoms() > 1:
                sele1 = CYS_pairs[0]
                sele2 = CYS_pairs[1]

                listOfAtomToCompare = cleanNumbers(findNeighbors(sele1, distA, sele2))
                if listOfAtomToCompare != []:
                    listOfAtomToCompare = sorted(listOfAtomToCompare, key=lambda x : x[-1])
                    minDistancePair = listOfAtomToCompare[0]
                    if minDistancePair[-1] < distA:
                        sele1_new = atoms.select('index ' + str(minDistancePair[0]) + ' and name '+str(minDistancePair[2]))
                        sele2_new = atoms.select('index ' + str(minDistancePair[1]) + ' and name '+str(minDistancePair[3]))
                        sele1_CB = atoms.select('resname CYS and name CB and resid ' + str(sele1_new.getResnums()[0])+
                            ' and chain ' + str(sele1_new.getChids()[0]))
                        sele2_CB = atoms.select('resname CYS and name CB and resid ' + str(sele2_new.getResnums()[0])+
                            ' and chain ' + str(sele2_new.getChids()[0]))
                        diheAng = calcDihedral(sele1_CB, sele1_new, sele2_new, sele2_CB)
                        DisulfideBonds_list.append([
                            sele1_new.getResnames()[0], sele1_new.getResnums()[0], sele1_new.getIcodes()[0], # resname, resnum, icode
                            minDistancePair[2], minDistancePair[0], sele1_new.getChids()[0], # atomname, atomindex, chain
                            sele2_new.getResnames()[0], sele2_new.getResnums()[0], sele2_new.getIcodes()[0], # resname, resnum, icode
                            minDistancePair[3], minDistancePair[1], sele2_new.getChids()[0], # atomname, atomindex, chain
                            round(minDistancePair[-1],4), round(float(diheAng),4) # distance, dihedral angle
                        ])
    except:
        atoms_SG = atoms.select('protein and resname CYS and name SG')
        if atoms_SG is None:
            LOGGER.info('Lack of cysteines (SG) in the structure.')
            DisulfideBonds_list = []

    DisulfideBonds_list_final = removeDuplicates(DisulfideBonds_list)

    sel_kwargs = {k: v for k, v in kwargs.items() if k.startswith('selection')}
    DisulfideBonds_list_final2 = filterInteractions(DisulfideBonds_list_final, atoms, **sel_kwargs)

    for kk in DisulfideBonds_list_final2:
        LOGGER.info("%3s%-5s%-1s%2s%5s%-6s  <---> %3s%-5s%-1s%2s%5s%-6s%8.1f%8.1f" % (kk[0], kk[1], kk[2], kk[5], kk[3], kk[4],
                                                            kk[6], kk[7], kk[8], kk[11], kk[9], kk[10], kk[12], kk[13]))

    LOGGER.info("Number of detected disulfide bonds: {0}.".format(len(DisulfideBonds_list_final2)))

    return DisulfideBonds_list_final2