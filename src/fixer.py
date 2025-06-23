from uuid import uuid1
import os
import subprocess
import numpy as np

from openmm.app.element import hydrogen
import openmm.app as app
from prody import LOGGER, parsePDB

from sklearn.cluster import KMeans
from .pdbfixer.pdbfixer import PDBFixer
from .pdbfixer.pdbfile import PDBFile
from .utils.settings import three2one, one2three, FIX_PDB_DIR, RAW_PDB_DIR, ROOT_DIR
from .download import fetchPDB, fetchPDB_BiologicalAssembly, fetchAF2

__all__ = ['LociFixer', 'fixPDB', 'createMutationfile', 'buildNE1']

MAX_NUM_RESIDUES = 20000

class LociFixer(PDBFixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _checkNumCalphas(self):
        n_ca = self.topology.getNumResidues()
        if n_ca > MAX_NUM_RESIDUES:
            m = f'Too many C-alphas: {n_ca}. Max. allowed: {MAX_NUM_RESIDUES}'
            raise RuntimeError(m)

    def _to_delete(self, max_loop=12):
        to_delete = []
        chains = list(self.topology.chains())
        for (chain_index, residue_index), missing_residues in self.missingResidues.items():
            chain = chains[chain_index]
            residues = list(chain.residues())
            if residue_index < len(residues):  
                residue_id = residues[residue_index].id
            else:
                residue_id = int(residues[-1].id) + 1
            if (
                residue_index == 0 # C terminal
                or residue_index == len(residues) # N terminal
                or len(missing_residues) > max_loop # Loop > 12 missing residues
                or int(residue_id) - len(missing_residues) < 0 # If negative residue_id, we remove the missing residues
            ):
                to_delete.append((chain_index, residue_index))
                continue
            # Remove non-standard missing residues (e.g. 'FGL' in 1AUK)
            # self.templates: ['VAL', 'DA', 'NME', 'TYR', 'ALA', 'DC', 'MET', 'ARG',
            # 'GLY', 'HIS', 'DT', 'GLU', 'PRO', 'LYS', 'ASP', 'ASN', 'TRP', 'PHE',
            # 'C', 'A', 'ILE', 'G', 'THR', 'U', 'SER', 'DG', 'GLN', 'CYS', 'ACE', 'LEU']
            self.missingResidues[(chain_index, residue_index)] = [
                res for res in missing_residues if res in self.templates
            ]
        # Now delete the entries after iteration
        for key in to_delete:
            del self.missingResidues[key]

    def fix(self, fix_loop=True, replaceNonstandard=True, keepElement=[]):
        if replaceNonstandard:
            self.findNonstandardResidues()
            self.replaceNonstandardResidues()
        self.removeHeterogens(keepElement=keepElement)

        # Check if there are missing atoms
        self.findMissingResidues()
        self.findMissingAtoms()

        # Restrict protein size
        try:
            self._checkNumCalphas()
        except RuntimeError as e:
            LOGGER.warn(f'No fix needed {str(e)}')
            return

        # Only CA atoms
        if self.topology.getNumResidues() == self.topology.getNumAtoms():
            LOGGER.warn('Only CA atoms found, no fix needed')
            return

        # Only fix the loop ≤ 12 residues
        max_loop = 12
        if fix_loop:
            # Remove missing residues > 12
            self._to_delete(max_loop=max_loop)
        else:
            to_delete = []
            for res, atoms in self.missingAtoms.items():
                if res.name != 'CYS':
                    to_delete.append(res)
                else:
                    for atom in atoms:
                        # Remove not SG atoms from f.missingAtoms[res]
                        if atom.name != 'SG':
                            self.missingAtoms[res].remove(atom)
            # Now delete the entries after iteration
            for res in to_delete:
                del self.missingAtoms[res]
            self.missingAtoms = {}
            self.missingResidues = {}
            self.missingTerminals = {}

        # Restrict number of missing residues to be less than 500
        nMissingResidues = sum(len(res) for res in self.missingResidues.values())
        while nMissingResidues > 500:
        # if nMissingResidues > 500:
            LOGGER.warn(f'Too many missing residues: {nMissingResidues}. Max. allowed: 500')
            LOGGER.info(f"Reduce missing loop from {max_loop} to {max_loop-2}")
            max_loop -= 2

            if max_loop < 0:
                LOGGER.warn(f"Too many missing residues: {nMissingResidues}. No fix needed")
                self.missingResidues = {}
                break
            # Remove missing residues > max_loop
            self._to_delete(max_loop=max_loop)
            nMissingResidues = sum(len(res) for res in self.missingResidues.values())

        # If >2000 res for 10,000 res system, we do not fix
        if len(self.missingAtoms) > 0.2 * self.topology.getNumResidues():
            LOGGER.warn(
                f"Too many residues with missing atoms {len(self.missingAtoms)} "
                f"out of {self.topology.getNumResidues()}. No fix needed")
            to_delete = []
            for res, atoms in self.missingAtoms.items():
                if res.name != 'CYS':
                    to_delete.append(res)
                else:
                    for atom in atoms:
                        # Remove not SG atoms from f.missingAtoms[res]
                        if atom.name != 'SG':
                            self.missingAtoms[res].remove(atom)
            # Now delete the entries after iteration
            for res in to_delete:
                del self.missingAtoms[res]
            self.missingAtoms = {}
            self.missingTerminals = {}
            self.addMissingAtoms()


        # We only fix if N<500 or %<20% of residues have missing atoms
        # if nMissingResidues < 500:
        #     self.addMissingAtoms()
        #     return
        # If >2000 res for 10,000 res system, we do not fix
        # elif len(self.missingAtoms) > 0.2 * self.topology.getNumResidues():
        #     LOGGER.warn(
        #         f"Too many residues with missing atoms {len(self.missingAtoms)} "
        #         f"out of {self.topology.getNumResidues()}. No fix needed")
        #     to_delete = []
        #     for res, atoms in self.missingAtoms.items():
        #         if res.name != 'CYS':
        #             to_delete.append(res)
        #         else:
        #             for atom in atoms:
        #                 # Remove not SG atoms from f.missingAtoms[res]
        #                 if atom.name != 'SG':
        #                     self.missingAtoms[res].remove(atom)
        #     # Now delete the entries after iteration
        #     for res in to_delete:
        #         del self.missingAtoms[res]
        #     self.missingAtoms = {}
        #     self.missingTerminals = {}
        #     self.addMissingAtoms()
        #     return
    
    def saveTopology(self, savePath, keepIds=True, modify_chain=None):
        """Save topology to a pdb file

        In our setting, we reset the chainID using ```chainName = chr(ord('A')+chainIndex%26)```
        """
        with open(savePath, 'w') as out_f:
            out_f.write('HEADER\n')
            PDBFile.writeFile(self.topology, self.positions, out_f, keepIds=keepIds, modify_chain=modify_chain)

def fixPDB(pdb, format='asu', 
           fix_loop=True, replaceNonstandard=True, refresh=False, folder='.'):
    """pdb could be a PDB file or a PDB ID"""
    os.makedirs(folder, exist_ok=True)
    if format == 'custom' and os.path.isfile(pdb):
        # Take the filename from the custom_PDB
        filename = pdb.split('/')[-1].split('.')[0]
        out = os.path.join(folder, f'{filename}.pdb')
        f = LociFixer(pdb)
        # Check pdb has DUM atoms
        has_dum_atom = any(atom.residue.name == 'DUM' for atom in f.topology.atoms())
        if has_dum_atom:
            f.fix(fix_loop=fix_loop, replaceNonstandard=replaceNonstandard, keepElement=['DUM'])
            f.saveTopology(savePath=out)
            out = buildNE1(out, folder=folder) # Build NE1 model
        else:
            f.fix(fix_loop=fix_loop, replaceNonstandard=replaceNonstandard)
            f.saveTopology(savePath=out)

    elif format == 'opm':
        out = os.path.join(folder, f'{pdb}-ne1.pdb')
        if os.path.isfile(out) and not refresh:
            LOGGER.info(f"File {out} already exists")
            return out
        pdbpath = fetchPDB(pdb, format=format, refresh=refresh, folder=RAW_PDB_DIR)
        f = LociFixer(pdbpath)
        f.fix(fix_loop=fix_loop, replaceNonstandard=replaceNonstandard, keepElement=['DUM'])
        opm_path = os.path.join(folder, f'{pdb}-opm.pdb')
        f.saveTopology(savePath=opm_path)
        out = buildNE1(opm_path, folder=folder, filename=pdb)

    elif format == 'asu':
        out = os.path.join(folder, f'{pdb}.pdb')
        if os.path.isfile(out) and not refresh:
            LOGGER.info(f"File {out} already exists")
            return out
        pdbpath = fetchPDB(pdb, format='pdb', refresh=refresh, folder=RAW_PDB_DIR)
        f = LociFixer(pdbpath)
        f.fix(fix_loop=fix_loop, replaceNonstandard=replaceNonstandard)
        f.saveTopology(savePath=out)

    elif format == 'af':
        out = os.path.join(folder, f'{pdb}.pdb')
        acc = pdb.split('-')[1]
        if os.path.isfile(out) and not refresh:
            LOGGER.info(f"File {out} already exists")
            return out
        pdbpath = fetchAF2(acc, refresh=refresh, folder=RAW_PDB_DIR)
        return pdbpath
    
    else: # format == bas*
        assemblyID = int(format[3:])
        out = os.path.join(folder, f'{pdb}-{assemblyID}.pdb')
        if os.path.isfile(out) and not refresh:
            LOGGER.info(f"File {out} already exists")
            return out
        pdbpath = fetchPDB_BiologicalAssembly(pdb, assemblyID, format='cif', 
                                        refresh=refresh, folder=RAW_PDB_DIR)
        f = LociFixer(pdbpath)
        f.fix(fix_loop=fix_loop, replaceNonstandard=replaceNonstandard)
        f.saveTopology(savePath=out)
    LOGGER.info(f"Fixed PDB file {out}")
    return out

def createMutationfile(pdbpath, chid, mutation, out=None):
    LOGGER.info(f"Creating mutation file for {mutation} in {pdbpath}")
    f = LociFixer(pdbpath)
    wt_aa = mutation[0]
    mut_aa = mutation[-1]
    resid = int(mutation[1:-1])
    # Find the residue to mutate.
    residue_map = dict()
    for chain in f.topology.chains():
        if chain.id == chid:
            for residue in chain.residues():
                if int(residue.id) == resid and three2one[residue.name] == wt_aa:
                    residue_map[residue] = one2three[mut_aa]
                    break
    # Find template residue for mutation.
    if len(residue_map) < 1:
        raise(ValueError("Cannot find the specified residue in the PDB file!"))
    try:
        template = f.templates[one2three[mut_aa]]
    except KeyError:
        raise(KeyError("Cannot find residue %s in template library!" % one2three[mut_aa]))
    # Find atoms to delete and atoms to add.
    deleteAtoms = [] # list of atoms to delete
    # Find atoms that should be deleted.
    for residue in residue_map.keys():
        replaceWith = residue_map[residue]
        residue.name = replaceWith
        template = f.templates[replaceWith]
        standardAtoms = set(atom.name for atom in template.topology.atoms())
        for atom in residue.atoms():
            if atom.element in (None, hydrogen) or atom.name not in standardAtoms:
                deleteAtoms.append(atom)
    # Delete atoms queued to be deleted.
    modeller = app.Modeller(f.topology, f.positions)
    modeller.delete(deleteAtoms)
    f.topology = modeller.topology
    f.positions = modeller.positions
    # Find missing atoms for the mutated residues and add them.
    f.findMissingResidues()
    f.findMissingAtoms()
    try:
        # Find the mutated residue. --> Only keep missing atoms of mutated residue
        for chain in f.topology.chains():
            if chain.id == chid:
                for residue in chain.residues():
                    if int(residue.id) == resid and three2one[residue.name] == mut_aa:
                        f.missingAtoms = {residue: f.missingAtoms[residue]}
                        break
                if residue in f.missingTerminals:
                    f.missingTerminals = {residue: f.missingTerminals[residue]}
                else:
                    f.missingTerminals = {}
                break
    except:
        pass
    f.addMissingAtoms()
    # Save the mutated PDB file.
    if out is None:
        pdbpath = os.path.abspath(pdbpath)
        pdbid = pdbpath.split('/')[-1][:4]
        folder = '/'.join(pdbpath.split('/')[:-1])
        out = f'{folder}/{pdbid}_{chid}_{mutation}.pdb'
    with open(out, 'w') as out_f:
        out_f.write('HEADER\n')
        PDBFile.writeFile(f.topology, f.positions, out_f, keepIds=True)
    return out

def execCGmembrane(opm_file, folder='.', filename=None,
                   radius_node=3.1, lower_thick=-16.6, upper_thick=16.6, radius_membrane=55, rr=15):
    """Run cgmembrane command line tool to build NE1 model.
    """
    exANM = os.path.join(ROOT_DIR, 'src/features/bin/cgmembrane')
    if not os.path.isfile(exANM):
        raise FileNotFoundError(f"cgmembrane executable not found at {exANM}")
    command = f"{exANM} {opm_file} -s {radius_node} -b {lower_thick} {upper_thick} -r {radius_membrane}"
    try:
        out = subprocess.run(command, shell=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        LOGGER.warn(f"execCGmembrane: command failed with error: {e}")
        return
    # Store output in a variable
    cgmembrane_output = out.stdout
    cgmembrane_output = cgmembrane_output.split('\n')

    NE1_atoms = [line for line in cgmembrane_output if line.startswith('ATOM')]
    NE1_atoms = [
        line for line in cgmembrane_output 
            if line.startswith('ATOM') and
            float(line[30:38])**2 + float(line[38:46])**2 > rr**2]
    # Format the filtered atoms into PDB format
    NE1_atoms = ["{}{:6.2f}{:6.2f}{:>12}\n".format(line, 1, 1, "M") for line in NE1_atoms]

    # Write the NE1 atoms to a file
    if filename is not None:
        out = os.path.join(folder, f'{filename}-ne1.pdb')
    else:
        out = os.path.join(folder, 'ne1.pdb')
    with open(out, 'w') as f:
        f.writelines(NE1_atoms)
    LOGGER.info(f"NE1 atoms written to {out}")
    return out


def buildCGmembrane(opm_file):
    """Build CGmembrane model from OPM file using cgmembrane.
    opm_file: must contain DUM


    src/features/bin/cgmembrane data/GJB2/structures/8qa2_opm_25Apr03.pdb -s 3.1 -b -98.3 -64.3 -r 55 > 98.pdb
    src/features/bin/cgmembrane data/GJB2/structures/8qa2_opm_25Apr03.pdb -s 3.1 -b -16.6 16.6 -r 55 > 16.pdb
    """
    try:
        pdb = parsePDB(opm_file)
    except FileNotFoundError as e:
        raise FileNotFoundError('Error: Cannot parsePDB opm_file')
    # check if DUM is present
    dumAtoms = pdb.select('resname DUM')
    if dumAtoms is None:
        raise ValueError('No DUM atoms found in the input file')
    
    # extract protein lines
    with open(opm_file, 'r') as f:
        lines = f.readlines()

    
    protein_lines = [line for line in lines
                    if "   DUM" not in line and 
                    (line.startswith('ATOM') or line.startswith('HETATM'))]
    
    # Locate z-coordinates of DUM atoms
    z_coords = dumAtoms.getCoords()[:, 2]
    # # Cluster z-positions to detect layer centers
    # kmeans = KMeans(n_clusters=2, random_state=42).fit(z_coords.reshape(-1, 1))
    # centers = np.sort(kmeans.cluster_centers_.flatten())
    # distance = abs(centers[1] - centers[0])
    # # if distance < 45: 
    # #     LOGGER.info("Detect: ONE membrane bilayer")
    # #     lower_thick = centers[0]
    # #     upper_thick = centers[1]
    # #     NE1_atoms = execCGmembrane(radius_node=radius_node, lower_thick=lower_thick,
    # #                                upper_thick=upper_thick, radius_membrane=radius_membrane)
    # #     NE1_atoms = [
    # #         line for line in NE1_atoms 
    # #             if line.startswith('ATOM')]
    # #     # Filter atoms based on x^2 + y^2 > RR^2 condition
    # #     # NE1_atoms = [
    # #     #     line for line in NE1_atoms 
    # #     #         if line.startswith('ATOM') and
    # #     #         float(line[30:38])**2 + float(line[38:46])**2 > rr**2]
    # # else:
    # #     LOGGER.info("Detect: TWO membrane bilayers")
        





def buildNE1(opm_file, folder='.', filename=None, radius_node=3.1, thick=16.6,
             rr=15, radius_membrane=55, remove=True):
    """Build NE1 model from OPM file using cgmembrane.

    opm_file: must contain DUM
    ne1_file: output file
    folder: output folder
    radius_node: radius of sphere of membrane particle (A)
        If radius_node is 3.1, two adjacent membrane particles are 6.2 A apart
    thick: bilayer thickness (A)
        If thick is 15.7, membrane thickness is 15.7*2 = 31.4 A (going from -thick to thick)
    rr: radius center to remove (A)
        If rr is 15, remove all atoms with x^2 + y^2 < rr^2
    radius_membrane: radius of membrane (A)
        If radius_membrane is 55, membrane radius is 55 A

    buildNE1('/mnt/nas_1/YangLab/loci/tandem/data/GJB2/structures/8qa2_opm_25Apr03.pdb',
        folder='/mnt/nas_1/YangLab/loci/tandem/data/GJB2/structures/',
        filename='8qa2_opm_25Apr03-',
        radius_node=3.1, thick=16.6, rr=15, radius_membrane=55, remove=False)
    
        
    """
    try:
        f = open(opm_file, 'r')
        lines = f.readlines()
        f.close()
    except FileNotFoundError:
        raise FileNotFoundError('Error: input file not found')
    # filename = os.path.basename(opm_file)
    if filename is None:
        filename = opm_file.split('/')[-1].split('.')[0]
    ne1_file = os.path.join(folder, f'{filename}-ne1.pdb')
    # check if DUM is present
    for line in lines:
        if 'DUM' in line and 'HETATM' in line:
            break
    else:
        raise ValueError('Error: input file does not contain DUM')
    # extract protein lines
    protein_lines = [line for line in lines
                        if "   DUM" not in line and 
                        (line.startswith('ATOM') or line.startswith('HETATM'))]
    # write protein files
    tempName = str(uuid1())
    protein = os.path.join(folder, f'{tempName}_protein.pdb')
    with open(protein, 'w') as f:
        f.writelines(protein_lines)
    # Run cgmembrane
    exANM = './src/features/bin/cgmembrane'
    command = f"{exANM} {protein} -s {radius_node} -b -{thick} {thick} -r {radius_membrane}"
    try:
        out = subprocess.run(command, shell=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        LOGGER.warn(f"buildNE1: command failed with error: {e}")
        return
    os.remove(protein)
    # Store output in a variable
    ne1_output = out.stdout
    ne1_output = ne1_output.split('\n')

    # Filter atoms based on x^2 + y^2 > RR^2 condition
    ne1_atoms = [
        line for line in ne1_output 
            if line.startswith('ATOM') and
            float(line[30:38])**2 + float(line[38:46])**2 > rr**2]
    # Format the filtered atoms into PDB format
    ne1_atoms = ["{}{:6.2f}{:6.2f}{:>12}\n".format(line, 1, 1, "M")
                 for line in ne1_atoms]
    full = protein_lines + ne1_atoms
    with open(ne1_file, 'w') as f:
        f.writelines(full)
    if remove:
        os.remove(opm_file)
    return ne1_file
