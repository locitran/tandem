# -*- coding: utf-8 -*-
"""This module defines a class and relative functions for mapping Uniprot
sequences to PDB and Pfam databases."""

import os
import re
import dill as pickle
import datetime
import time
import numpy as np
import urllib.parse
import requests 
import re
import traceback

import prody
from prody import parsePDB, Atomic
from prody.utilities import openURL
from Bio.pairwise2 import align as bioalign
from Bio.pairwise2 import format_alignment

from ..utils.logger import LOGGER
from ..download import fetchPDB, fetchPDB_BiologicalAssembly, fetchAF2
from ..utils.settings import RAW_PDB_DIR, one2three

__author__ = "Luca Ponzoni"
__date__ = "December 2019"
__maintainer__ = "Luca Ponzoni"
__email__ = "lponzoni@pitt.edu"
__status__ = "Production"

# Modified by Loci Tran
__secondary_author__ = "Loci Tran"

__all__ = ['queryUniprot', 'UniprotMapping', 'mapSAVs2PDB']

def queryUniprot(*args, n_attempts=3, dt=1, **kwargs):
    """
    Redefine prody function to check for no internet connection
    """
    attempt = 0
    while attempt < n_attempts:
        try:
            _ = openURL('http://www.uniprot.org/')
            break
        except:
            LOGGER.info(
                f'Attempt {attempt} to contact www.uniprot.org failed')
            attempt += 1
            time.sleep((attempt+1)*dt)
    else:
        _ = openURL('http://www.uniprot.org/')
    return prody.queryUniprot(*args, **kwargs)

def verifyAF(pdbpath):
    """Check if the PDB file is an AlphaFold structure."""
    pdbpath = os.path.abspath(pdbpath)
    title = os.path.basename(pdbpath)
    if title.startswith('AF-'):
        return True
    if os.path.isfile(pdbpath):
        with open(pdbpath, 'r') as f:
            first_line = f.readline()
            if 'alphafoldserver.com/output-terms' in first_line:
                return True
    return False

class UniprotMapping:

    def __init__(self, acc, recover_pickle=False, **kwargs):
        self.acc = self._checkAccessionNumber(acc)
        self.uniq_acc = None
        self.fullRecord = None
        self.sequence = None
        self.sequence_length = None
        self.PDBrecords = None
        self.PDBmappings = None
        self.AF2mapping = None
        self.customPDBmapping = None
        self._align_algo_args = None
        self._align_algo_kwargs = None
        self.timestamp = None
        self.Pfam = None
        self.AF2 = None
        self.feats = {}
        self._refresh = not(recover_pickle) # if recover_pickle is True, refresh is False
        assert type(recover_pickle) is bool
        if recover_pickle:
            try:
                self.recoverPickle(**kwargs)
            except Exception:
                LOGGER.warn(f'Unable to recover pickle: UniprotMap-{acc}.pkl')
                self.refresh()
        else:
            self.refresh()

    def refresh(self):
        """Refresh imported Uniprot records and mappings, and
        delete precomputed alignments.
        """
        # import Uniprot record and official accession number
        self.fullRecord = queryUniprot(self.acc)
        self.uniq_acc = self.fullRecord['accession   0']
        # import main sequence and PDB records
        rec = self.fullRecord
        self.sequence = rec['sequence   0'].replace("\n", "")
        self.sequence_length = len(self.sequence)
        self.PDBrecords = [rec[key] for key in rec.keys()
                           if key.startswith('dbRef') and 'PDB' in rec[key]]
        # parse PDB records into PDB mappings, easier to access
        self._initiatePDBmappings()
        # set remaining attributes
        self.customPDBmapping = {}
        self._align_algo_args = ['localxs', -0.5, -0.1]
        self._align_algo_kwargs = {'one_alignment_only': True}
        self.timestamp = str(datetime.datetime.utcnow())
        self.AF2 = [rec[key][1][1] for key in rec.keys()
                    if key.startswith('dbRef') and ('type', 'AlphaFoldDB') in rec[key]]
        if len(self.AF2) > 0:
            self.AF2 = self.AF2[0]
            self._initiateAF2mapping()
        return

    def getPDBmappings(self, PDBID=None):
        """Returns a list of dictionaries, with mappings of the Uniprot
        sequence onto single PDB chains. For each PDB chain, the residue
        intervals retrieved from the Uniprot database are parsed into a list
        of tuples ('chain_sel') corresponding to endpoints of individual
        segments. NB: '@' stands for 'all chains', following Uniprot naming
        convention.
        """
        if PDBID is None:
            return self.PDBmappings
        # retrieve record for given PDBID
        PDBID = PDBID.upper()
        recs = [d for d in self.PDBmappings if d['PDB'] == PDBID]
        # there should be only one record for a given PDBID
        if len(recs) == 0:
            raise ValueError(f'PDBID {PDBID} not found in Uniprot record.')
        if len(recs) > 1:
            m = f"Multiple entries in Uniprot record for PDBID {PDBID}. "
            m += "Only the first one will be considered."
            LOGGER.warn(m)
        return recs[0]

    def alignSinglePDB(self, PDBID, chain='longest'):
        """Aligns the Uniprot sequence with the sequence from the given PDB entry."""
        PDBrecord = self.getPDBmappings(PDBID)
        if PDBrecord['chain_seq'] is None:
            raise RuntimeError("Unable to parse PDB.")
        # retrieve chain mappings. Format: {'A': [(1, 10), (15, 100)]}
        mappings = PDBrecord['chain_sel']
        # retrieve list of chains from Uniprot record for given PDBID
        all_chains = set(mappings.keys())
        if '@' in all_chains:
            all_chains = PDBrecord['chain_seq'].keys()
        # select chains to be aligned
        chains_to_align = []
        if chain == 'longest':
            # align only the longest chain in the PDB file
            nCA_max = 0
            for c in sorted(all_chains):
                nCA = len(PDBrecord['chain_res'][c])
                if nCA > nCA_max:
                    nCA_max = nCA
                    chains_to_align = [c]
        elif chain == 'all' or chain == '@':
            # align all chains
            chains_to_align = list(all_chains)
        elif chain in all_chains:
            # align only the requested chain
            chains_to_align = [chain]
        else:
            raise ValueError(f'chain {chain} not found in Uniprot record.')
        # align selected chains with BioPython module pairwise2
        self._calcAlignments(PDBID, chains_to_align)
        # return alignments and maps of selected chains
        rec = [d for d in self.PDBmappings if d['PDB'] == PDBID][0]
        sel_alignms = {c: rec['alignments'][c] for c in chains_to_align}
        sel_maps = {c: rec['maps'][c] for c in chains_to_align}
        return sel_alignms, sel_maps

    def alignCustomPDB(self, PDB, title=None, folder=RAW_PDB_DIR):
        """Aligns the Uniprot sequence with the sequence from the given PDB.
        """
        assert isinstance(PDB, (str, Atomic)), \
            'PDB must be a PDBID or an Atomic instance (e.g. AtomGroup).'
        if isinstance(PDB, str):
            try:
                if os.path.isfile(PDB):
                    pdb = parsePDB(PDB, model=1)
                else:
                    # PDB is a PDBID
                    LOGGER.info(f'Fetching PDB {PDB}...')
                    pdbpath = fetchPDB(PDB, format='pdb', folder=folder, refresh=self._refresh)
                    if pdbpath is not None:
                        pdb = parsePDB(pdbpath, model=1)
            except Exception as e:
                msg = (
                    'Unable to import structure: PDB ID might be invalid'
                    ' or PDB file might be corrupted.\n'
                    f'PDB error: {e}')
                LOGGER.error(msg)
            if title is None:
                title = os.path.basename(PDB.strip())
                title = title.replace(' ', '_')
        else:
            if title is None:
                title = PDB.getTitle()

        alphafold = verifyAF(PDB)
        if alphafold:
            LOGGER.info(f'AlphaFold2 structure detected: {title}')
        LOGGER.info(f'Aligning {title}...')
        pdb = pdb.ca
        # Initilize
        customPDBmapping = {
            'PDB': title,
            'chain_res': {},
            'chain_seq': {},
            'chain_len': {},
            'warnings': [],
            'alphafold': alphafold,
            'confidence': {}
        }
        all_chains = set(pdb.getChids())
        chains_to_align = list(all_chains)
        # store resids and sequence of selected chains
        for c in chains_to_align:
            if c in customPDBmapping['chain_res']:
                continue
            chain = pdb.select(f'chain {c}')
            if chain:
                customPDBmapping['chain_res'][c] = chain.getResnums()
                customPDBmapping['chain_seq'][c] = chain.getSequence()
                customPDBmapping['chain_len'][c] = chain.getResnums().__len__() # chain.numResidues()
            else:
                customPDBmapping['warnings'].append(f'Chain {c} not found in PDB.')
            if alphafold:
                confidence = chain.getBetas()
                customPDBmapping['confidence'][c] = confidence

        self.customPDBmapping = customPDBmapping
        # align selected chains with BioPython module pairwise2
        self._calcCustomAlignments(chains_to_align)
        return customPDBmapping

    def alignAllPDBs(self, chain='longest'):
        """Aligns the Uniprot sequence with the sequences of all PDBs in the
        Uniprot record.
        """
        assert chain in ['longest', 'all']
        PDBIDs_list = [d['PDB'] for d in self.PDBmappings]
        for PDBID in PDBIDs_list:
            try:
                _ = self.alignSinglePDB(PDBID, chain=chain)
            except:
                continue
        return self.PDBmappings
    
    def mapMultipleResidues(self, resids, wt_aas):
        """Map multiple amino acids in a Uniprot sequence to PDBs.
        resids is a list of residue indices in the Uniprot sequence.
        wt_aas is a list of wild-type amino acids at the given positions.
        """
        resid_interval = (min(resids), max(resids))
        matches = []
        for PDBrecord in self.PDBmappings:
            PDBID = PDBrecord['PDB']
            chain_sel = PDBrecord['chain_sel']
            chain_seq = PDBrecord['chain_seq']
            resolution = PDBrecord['resolution']
            chain_len = PDBrecord['chain_len']
            if isinstance(resolution, str): # NMR -> assign 1
                resolution = 1
            if chain_sel is None:
                # add all chains anyway, if possible
                if chain_seq is not None:
                    chainIDs = PDBrecord['chain_seq'].keys()
                else:
                    chainIDs = []
                for chainID in chainIDs:
                    matches.append((PDBID, chainID, -999, -999, resolution))
            else:
                for chainID, intervals in chain_sel.items():
                    if None in intervals:
                        matches.append((PDBID, chainID, -999, -999, resolution)) # range is undefined, add it anyway
                    elif np.any([ # Check chain interval overlaps with the residue interval.
                        max(interval[0], resid_interval[0]) <= min(interval[1], resid_interval[1]) \
                        for interval in intervals
                    ]):
                        c_seq_len = sum([i[1]-i[0]+1 for i in intervals]) # defined as by UniProt website, column “Positions”
                        c_resolved_len = chain_len[chainID]
                        coverage_perc = c_resolved_len / self.sequence_length
                        if coverage_perc >= 0.5:
                            matches.append((PDBID, chainID, c_seq_len, c_resolved_len, coverage_perc, resolution))
            # sort first by c_resolved_len, resolution, then PDBID and chainID
            matches.sort(key=lambda x: (-x[2], -x[3], -x[4], x[5], x[0][::-1], x[1])) # Smallest score first

        # now align selected chains to find actual hits
        hits = np.zeros(len(resids), dtype=[
            ('>asu:PDB_coords', 'U100'),
            ('>asu:c_seq_len', 'i4'),
            ('>asu:c_resolved_len', 'i4'),
            ('>bas:PDB_coords', 'U100'),
            ('>opm:PDB_coords', 'U100'),
            ('resolution', 'f4'),
        ])

        # Loop over all residues
        for idx, (resid, wt_aa) in enumerate(zip(resids, wt_aas)):
            # Skip if a given resid is not within uniprot sequence.
            if not (1 <= resid <= len(self.sequence)):
                hits[idx]['>asu:PDB_coords'] = f'Cannot map, resid {resid} out of range {self.sequence_length}'
                continue
            # Skip if a given wt_aa is not the same as the residue in uniprot sequence.
            u_aa = self.sequence[resid-1]
            if wt_aa != u_aa:
                hits[idx]['>asu:PDB_coords'] = f'Cannot map, wt residue is {u_aa} not {wt_aa}'
                continue
            # Loop over all PDBs: coverage_perc >= 0.5
            for PDBID, chainID, c_seq_len, c_resolved_len, coverage_perc, resolution in matches:
                try:
                    als, maps = self.alignSinglePDB(PDBID, chain=chainID)
                    # {'A': {uniprot_resid: (PDB_resid, aa), ...}, ...}
                except:
                    continue
                if chainID == '@':
                    c_list = sorted(maps.keys())
                else:
                    c_list = [chainID]
                for c in c_list:
                    hit = maps[c].get(resid) # maps = {c: rec['maps'][c] for c in chains_to_align}
                    if hit is None:
                        continue
                    elif u_aa is not None and hit[1] != u_aa:
                        continue
                    else:
                        if c.strip() == '':
                            c = '?'
                        # pdbID / chainID / resID / resName
                        res_map = f'{PDBID} {c} {hit[0]} {hit[1]}'
                        hits[idx] = (res_map, c_seq_len, c_resolved_len, '', '', resolution)
                        break
                if len(hits[idx]['>asu:PDB_coords']) > 0:
                    break
            
            if len(hits[idx]['>asu:PDB_coords']) == 0:
                hits[idx]['>asu:PDB_coords'] = 'Cannot map, no hits found'
                # Coverage percentage, 𝑟=𝑠/𝑙 
                # If searching all PDB with coverage_perc >= 50 results no hit
                # Take AlphaFold2 structure
                if self.AF2mapping is None:
                    continue
                chid = self.AF2mapping['chain']
                title = self.AF2mapping['AF2']
                # If residue is in very low confidence region, skip
                confidence = self.AF2mapping['confidence'][resid-1]
                if confidence < 50:
                    hits[idx]['>asu:PDB_coords'] = f'Cannot map, very low confidence region {confidence}'
                    continue
                res_map = f'{title} {chid} {resid} {u_aa}'
                hits[idx] = (res_map, self.AF2mapping['chain_len'], self.AF2mapping['chain_len'],
                                'Cannot map', 'Cannot map', -999)

        # Find corresponding OPM and Assembly PDBs
        asu_coords = hits['>asu:PDB_coords']
        res_mappings = self.mapOPMorAssembly(asu_coords)
        for i, res_map in enumerate(res_mappings):
            hits[i]['>bas:PDB_coords'] = res_map['>bas:PDB_coords']
            hits[i]['>opm:PDB_coords'] = res_map['>opm:PDB_coords']
        return hits

    def mapOPMorAssembly(self, asu_coords, folder=RAW_PDB_DIR):
        """Map a hit from mapSingleResidue to the corresponding OPM or
        assembly PDB.
        """
        res_mappings = np.zeros(len(asu_coords), dtype=[
            ('>bas:PDB_coords', 'U100'),
            ('>opm:PDB_coords', 'U100'),
        ])
        # Group hits by PDBID
        groups = {}
        for i, asu_map in enumerate(asu_coords):
            PDBID = asu_map.split()[0]
            if "Cannot map" in asu_map or len(asu_map.split()) != 4:
                res_mappings[i]['>opm:PDB_coords'] = "Cannot map"
                res_mappings[i]['>bas:PDB_coords'] = "Cannot map"
                continue
            if PDBID not in groups:
                groups[PDBID] = {'asu_map': [], 'index': []}
            groups[PDBID]['asu_map'].append(asu_map)
            groups[PDBID]['index'].append(i)

        # Process each group
        for PDBID, group in groups.items():
            # Parse GraphQL for PDB
            LOGGER.info(f'Find {PDBID} OPM and Assembly...')
            assemblies, opm = self._parseGraphQLforPDB(PDBID)
            g_asu_maps, g_indices = group.values()
            if opm:
                opmPath = fetchPDB(PDBID, format='opm', folder=folder, refresh=self._refresh)
                pdb = prody.parsePDB(opmPath)
                # Loop over all coord
                for map_idx, asu_map in zip(g_indices, g_asu_maps):
                    chid, resid, aa = asu_map.split()[1:]
                    # selres = pdb.ca[(chid, int(resid), '')]
                    selres = pdb.ca.select(f'chain {chid} and resnum {resid} and resname {one2three[aa]}')
                    if selres is None: 
                        res_mappings[map_idx]['>opm:PDB_coords'] = 'Cannot map, selres is None'
                    else:
                        n_atoms = selres.numAtoms()

                        if n_atoms > 1:
                            res_mappings[map_idx]['>opm:PDB_coords'] = 'Cannot map, >1 atom found'
                        else:
                            res_mappings[map_idx]['>opm:PDB_coords'] = asu_map
            else:
                for i in g_indices:
                    res_mappings[i]['>opm:PDB_coords'] = 'Cannot map, no opm info'

            if len(assemblies) == 0:
                for i in g_indices:
                    res_mappings[i]['>bas:PDB_coords'] = 'Cannot map, no biounit info'
                continue
            # Loop over all assemblies
            for assembly in assemblies:
                id = assembly.split('-')[1]
                basPath = fetchPDB_BiologicalAssembly(PDBID, id, format='cif', 
                                                      folder=folder, refresh=self._refresh)
                pdb = parsePDB(basPath)
                # Restrict protein size
                try:
                    self._checkNumCalphas(pdb)
                except Exception as e:
                    for i in g_indices:
                        res_mappings[i]['>bas:PDB_coords'] = f'Cannot map, {str(e)}'
                    continue
                for map_idx, asu_map in zip(g_indices, g_asu_maps):
                    # Correct format: PDBID / chainID / resID / resName / assemblyID
                    bas_coord_splitting = res_mappings[map_idx]['>bas:PDB_coords'].split()
                    if len(bas_coord_splitting) == 5 and bas_coord_splitting[0] == PDBID:
                        continue
                    _, chain, resid, aa = asu_map.split()
                    # selres = pdb.ca[(chain, int(resid), '')]
                    selres = pdb.ca.select(f'chain {chain} and resnum {resid} and resname {one2three[aa]}')
                    if selres is None:
                        res_mappings[map_idx]['>bas:PDB_coords'] = 'Cannot map, sel_res is None'
                    else:
                        n_atoms = selres.numAtoms()
                        if n_atoms > 1:
                            res_mappings[map_idx]['>bas:PDB_coords'] = f'Cannot map, >1 atom found {PDBID}-{id}'
                        else:
                            res_mappings[map_idx]['>bas:PDB_coords'] = f'{asu_map} {id}'
                                
                if np.any(['Cannot map' not in res_mappings[map_idx]['>bas:PDB_coords'] for i in g_indices]):
                    break
        return res_mappings

    def _checkNumCalphas(self, ag):
        MAX_NUM_RESIDUES = 18000
        n_ca = ag.ca.numAtoms()
        if n_ca > MAX_NUM_RESIDUES:
            m = f'Too many C-alphas: {n_ca}. Max. allowed: {MAX_NUM_RESIDUES}'
            raise RuntimeError(m)

    def _parseGraphQLforPDB(self, PDBID):
        """Parse the graphQL response to extract the assemblies and OPM info.
        More info at https://www.rcsb.org/docs/programmatic-access/web-apis-overview#data-api
        """
        query = (
            f'{{\n'
            f'  entry(entry_id: "{PDBID}") {{\n'
            f'    rcsb_id\n'
            f'    assemblies {{\n'
            f'      rcsb_assembly_container_identifiers {{\n'
            f'        rcsb_id\n'
            f'      }}\n'
            f'    }}\n'
            f'    polymer_entities {{\n'
            f'      rcsb_polymer_entity_annotation {{\n'
            f'        annotation_id\n'
            f'        assignment_version\n'
            f'        description\n'
            f'        name\n'
            f'        provenance_source\n'
            f'        type\n'
            f'      }}\n'
            f'      entity_poly {{\n'
            f'        nstd_linkage\n'
            f'        nstd_monomer\n'
            f'        pdbx_seq_one_letter_code\n'
            f'        pdbx_seq_one_letter_code_can\n'
            f'        pdbx_sequence_evidence_code\n'
            f'        pdbx_strand_id\n'
            f'        pdbx_target_identifier\n'
            f'        rcsb_artifact_monomer_count\n'
            f'        rcsb_conflict_count\n'
            f'        rcsb_deletion_count\n'
            f'        rcsb_entity_polymer_type\n'
            f'        rcsb_insertion_count\n'
            f'        rcsb_mutation_count\n'
            f'        rcsb_non_std_monomer_count\n'
            f'        rcsb_prd_id\n'
            f'        rcsb_sample_sequence_length\n'
            f'        type\n'
            f'      }}\n'
            f'    }}\n'
            f'  }}\n'
            f'}}'
        )
        encoded_query = urllib.parse.quote(query)
        url = f'https://data.rcsb.org/graphql?query={encoded_query}'

        # Parse the graphQL
        response = requests.get(url)
        data = response.json()
        if data['data']['entry'] is None:
            return [], False
        # Extract the assemblies and OPM info
        assemblies = [ele['rcsb_assembly_container_identifiers']['rcsb_id'] for ele in data['data']['entry']['assemblies']]
        opm = False
        for ele in data['data']['entry']['polymer_entities']:
            if opm:
                break
            if ele['rcsb_polymer_entity_annotation'] is None:
                break
            for entity in ele['rcsb_polymer_entity_annotation']:
                if entity['type'] == 'OPM':
                    opm = True
                    break
        return assemblies, opm # (e.g. ['1KV3-1', '1KV3-2', '1KV3-3'], False)
        
    def mapMultipleRes2CustomPDBs(self, resids, wt_aas):
        nSAVs = len(resids)
        hits = np.zeros(nSAVs, dtype=[
            ('>asu:PDB_coords', 'U100'),
            ('>asu:c_seq_len', 'i4'),
            ('>asu:c_resolved_len', 'i4'),
            ('>bas:PDB_coords', 'U100'),
            ('>opm:PDB_coords', 'U100'),
            ('resolution', 'f4'),
        ])

        # Sort chains by length: longest first
        customPDBmapping = self.customPDBmapping
        sorted_chains = sorted(customPDBmapping['chain_len'], 
                               key=lambda x: (-customPDBmapping['chain_len'][x], x))
        title = customPDBmapping['PDB']
        maps = customPDBmapping['maps'] 
        alphafold = customPDBmapping['alphafold']
        # keys: pdb chids
        # values: (PDBresids[resindx_PDB], aaC)
        # hit[0]: resid; hit[1]: aa ; e.g. 'A': (PDBresids[resindx_PDB], aaC)= 5038: (5037, 'S')
        for idx, (resid, wt_aa) in enumerate(zip(resids, wt_aas)):
            for c in sorted_chains:
                chain_len = customPDBmapping['chain_len'][c]
                if resid not in maps[c]:
                    continue
                hit = maps[c][resid]
                if hit[1] != wt_aa:
                    msg = 'Residue {} ({}) was found in chain {} '.format(resid, c, wt_aa)
                    msg += 'of PDB {} but has wrong aa, residue {} ({})'.format(title, hit[0], hit[1])
                    LOGGER.info(msg)
                    continue
                if not alphafold:
                    res_map = f'{title} {c} {hit[0]} {hit[1]}'
                    hits[idx] = (res_map, chain_len, chain_len, '', '', -999)
                    break
                else:
                    confidence = customPDBmapping['confidence'][c][hit[0]]
                    if confidence < 50:
                        hits[idx]['>asu:PDB_coords'] = f'Cannot map, very low confidence region {confidence}'
                        continue
                    res_map = f'{title} {c} {hit[0]} {hit[1]}'
                    hits[idx] = (res_map, chain_len, chain_len, '', '', -999)
            if len(hits[idx]['>asu:PDB_coords']) == 0:
                hits[idx]['>asu:PDB_coords'] = 'Cannot map, no hits found'

        # Find corresponding OPM and Assembly PDBs
        asu_coords = hits['>asu:PDB_coords']
        res_mappings = self.mapOPMorAssembly(asu_coords)
        for i, res_map in enumerate(res_mappings):
            hits[i]['>bas:PDB_coords'] = res_map['>bas:PDB_coords']
            hits[i]['>opm:PDB_coords'] = res_map['>opm:PDB_coords']
        return hits

    def savePickle(self, **kwargs):
        filename = kwargs.get('filename', self.acc)
        folder = kwargs.get('folder', '.')
        folder = os.path.join(folder, 'pickles/uniprot')
        os.makedirs(folder, exist_ok=True)
        filename = 'UniprotMap-' + filename + '.pkl'
        pickle_path = os.path.join(folder, filename)
        cache = self.customPDBmapping
        # save pickle
        pickle.dump(self, open(pickle_path, "wb"))
        self.customPDBmapping = cache
        LOGGER.info("Pickle '{}' saved.".format(pickle_path))
        return pickle_path

    def recoverPickle(self, days=30*6, **kwargs):
        folder = kwargs.get('folder', '.')
        folder = os.path.join(folder, 'pickles/uniprot')
        filename = kwargs.get('filename', self.acc)
        filename = 'UniprotMap-' + filename + '.pkl'
        pickle_path = os.path.join(folder, filename)
        # check if pickle exists
        if not os.path.isfile(pickle_path):
            raise IOError("{} not found".format(pickle_path))
        # load pickle
        recovered_self = pickle.load(open(pickle_path, "rb"))
        if self.acc not in [recovered_self.acc, recovered_self.uniq_acc]:
            raise ValueError('Accession number in recovered pickle (%s) '
                             % recovered_self.uniq_acc + 'does not match.')
        # check timestamp and ignore pickles that are too old
        date_format = "%Y-%m-%d %H:%M:%S.%f"
        t_old = datetime.datetime.strptime(recovered_self.timestamp,
                                           date_format)
        t_now = datetime.datetime.utcnow()
        Delta_t = datetime.timedelta(days=days)
        if t_old + Delta_t < t_now:
            raise RuntimeError(
                'Pickle {} was too old and was ignored.'.format(filename))
        self.fullRecord = recovered_self.fullRecord
        self.uniq_acc = recovered_self.uniq_acc
        self.sequence = recovered_self.sequence
        self.sequence_length = recovered_self.sequence_length
        self.PDBrecords = recovered_self.PDBrecords
        self.PDBmappings = recovered_self.PDBmappings
        self.customPDBmapping = recovered_self.customPDBmapping
        self._align_algo_args = recovered_self._align_algo_args
        self.AF2 = recovered_self.AF2
        self.AF2mapping = recovered_self.AF2mapping
        self._align_algo_kwargs = recovered_self._align_algo_kwargs
        self.timestamp = recovered_self.timestamp
        self.Pfam = recovered_self.Pfam
        self.feats = recovered_self.feats
        LOGGER.info("Pickle '{}' recovered.".format(pickle_path))
        return

    def resetTimestamp(self):
        self.timestamp = str(datetime.datetime.utcnow())

    def _checkAccessionNumber(self, acc):
        if '-' in acc:
            acc = acc.split('-')[0]
            message = 'Isoforms are not allowed, the main sequence for ' + \
                      acc + ' will be used instead.'
            # LOGGER.warn(message)
            print(message)
        return acc

    def _parseSelString(self, sel_str):
        # example: "A/B/C=15-100, D=30-200"
        # or: "@=10-200"
        parsedSelStr = {}
        for segment in sel_str.replace(' ', '').split(','):
            fields = segment.split('=')
            chains = fields[0].split('/')
            resids = fields[1].split('-')
            try:
                resids = tuple([int(s) for s in resids])
            except Exception:
                # sometimes the interval is undefined,
                # e.g. "A=-"
                resids = None
            for chain in chains:
                parsedSelStr.setdefault(chain, []).append(resids)
        return parsedSelStr

    def _initiateAF2mapping(self, folder=RAW_PDB_DIR, version=4):
        mappings = {}
        try:
            pdbpath = fetchAF2(self.AF2, folder=folder, version=version)
            if pdbpath is not None:
                pdb = parsePDB(pdbpath)
            else:
                raise ValueError('PDB file not found.')
            mappings['AF2'] = pdb.getTitle()
        except Exception as e:
            msg = "Error while parsing PDB: {}".format(e)
            LOGGER.warn(msg)
            return

        chids = set(pdb.getChids())
        if len(chids) != 1:
            msg = "Error: AF2 PDB file contains more than one chain."
            LOGGER.warn(msg)
            return
        else:
            chid = list(chids)[0]
        chain = pdb.select(f'chain {chid}')
        if chain is None:
            msg = "Error: AF2 PDB file contains no chain."
            LOGGER.warn(msg)
            return
        mappings['chain'] = chid
        mappings['chain_res'] = chain.ca.getResnums()
        mappings['chain_seq'] = chain.ca.getSequence()
        mappings['chain_len'] = chain.ca.getResnums().__len__()
        mappings['confidence'] = chain.ca.getBetas()
        self.AF2mapping = mappings
        return

    def _initiatePDBmappings(self, folder=RAW_PDB_DIR):
        illegal_chars = r"[^A-Za-z0-9-@=/,\s]"
        PDBmappings = []
        for singlePDBrecord in self.PDBrecords:
            PDBID = singlePDBrecord.get('PDB').upper()
            mapping = {'PDB': PDBID,
                       'chain_sel': {},
                       'chain_res': {},
                       'chain_seq': {},
                       'warnings': [],
                       'resolution': None,
                       'chain_len': {},
                       }
            # import selection string
            sel_str = singlePDBrecord.get('chains')
            if sel_str is None:
                mapping['warnings'].append('Empty selection string.')
            else:
                # check for illegal characters in selection string
                match = re.search(illegal_chars, sel_str)
                if match:
                    chars = re.findall(illegal_chars, sel_str)
                    message = "Illegal characters found in 'chains' " \
                              + 'selection string: ' + ' '.join(chars)
                    mapping['warnings'].append(message)
                else:
                    parsed_sel_str = self._parseSelString(sel_str)
                    mapping['chain_sel'] = parsed_sel_str

            # store resolution of PDB
            resolution = singlePDBrecord.get('resolution')
            if resolution is None:
                resolution = "NMR"
            else:
                resolution = float(resolution.split()[0])
            mapping['resolution'] = resolution
            # store resids and sequence of PDB chains
            try:
                pdbpath = fetchPDB(PDBID, format='pdb', folder=folder, refresh=self._refresh)
                if pdbpath is not None:
                    pdb = parsePDB(pdbpath, subset='calpha')
                else:
                    raise ValueError('PDB file not found.')
            except Exception:
                mapping['chain_sel'] = None
                mapping['chain_res'] = None
                mapping['chain_seq'] = None
                mapping['chain_len'] = None
                msg = "Error while parsing PDB: {}".format(e)
                mapping['warnings'].append(msg)
                LOGGER.warn(msg)
                PDBmappings.append(mapping)
                continue

            # Loop c in chain_sel, if c is not in set(pdb.getChids()), remove it from chain_sel
            for c in list(mapping['chain_sel']):
                if c not in set(pdb.getChids()):
                    mapping['chain_sel'].pop(c)
                    
            # Loop c in set(pdb.getChids()), if pdb[c] cannot be detected, remove it from chain_sel
            for c in set(pdb.getChids()):
                chain = pdb.select(f'chain {c}')
                if chain:
                    mapping['chain_res'][c] = chain.getResnums()
                    mapping['chain_seq'][c] = chain.getSequence()
                    mapping['chain_len'][c] = chain.getResnums().__len__()
                else:
                    if c in list(mapping['chain_sel']):
                        mapping['chain_sel'].pop(c)
                    msg = "Error while add chain {} info. to dictionary.".format(c)
                    mapping['warnings'].append(msg)
                    LOGGER.warn(msg)
            PDBmappings.append(mapping)
        self.PDBmappings = PDBmappings
        if PDBmappings == []:
            LOGGER.warn('No PDB entries have been found '
                        'that map to given sequence.')
        return

    ### Alignments

    def _align(self, seqU, seqC, PDBresids, print_info=False):
        algo = self._align_algo_args[0]
        args = self._align_algo_args[1:]
        kwargs = self._align_algo_kwargs
        # align Uniprot and PDB sequences
        al = None
        if algo == 'localxx':
            al = bioalign.localxx(seqU, seqC, *args, **kwargs)
        elif algo == 'localxs':
            al = bioalign.localxs(seqU, seqC, *args, **kwargs)
        else:
            al = bioalign.localds(seqU, seqC, *args, **kwargs)
        if print_info is True:
            info = format_alignment(*al[0])
            LOGGER.info(info[:-1])
            idnt = sum([1 for a1, a2 in zip(al[0][0], al[0][1]) if a1 == a2])
            frac = idnt/len(seqC)
            m = "{} out of {} ({:.1%}) residues".format(idnt, len(seqC), frac)
            m += " in the chain are identical to Uniprot amino acids."
            LOGGER.info(m)
        # compute mapping between Uniprot and PDB chain resids
        aligned_seqU = al[0][0]
        aligned_seqC = al[0][1]
        mp = {}
        resid_U = 0
        resindx_PDB = 0
        for i in range(len(aligned_seqU)):
            aaU = aligned_seqU[i]
            aaC = aligned_seqC[i]
            if aaU != '-':
                resid_U += 1
                if aaC != '-':
                    mp[resid_U] = (PDBresids[resindx_PDB], aaC)
            if aaC != '-':
                resindx_PDB += 1
        return al[0][:2], mp

    def _quickAlign(self, seqU, seqC, PDBresids):
        '''Works only if PDB sequence and resids perfectly match
        those found in Uniprot.'''
        s = ['-'] * len(seqU)
        mp = {}
        for resid, aaC in zip(PDBresids, seqC):
            indx = resid-1
            try:
                aaU = seqU[indx]
            except:
                raise RuntimeError('Invalid resid in PDB.')
            if resid in mp:
                raise RuntimeError('Duplicate resid in PDB.')
            elif aaC != aaU:
                raise RuntimeError('Non-WT aa in PDB sequence.')
            else:
                mp[resid] = (resid, aaC)
                s[indx] = aaC
        aligned_seqC = "".join(s)
        return (seqU, aligned_seqC), mp

    def _calcAlignments(self, PDBID, chains_to_align):
        seqUniprot = self.sequence
        PDBrecord = self.getPDBmappings(PDBID)
        alignments = PDBrecord.setdefault('alignments', {})
        maps = PDBrecord.setdefault('maps', {})
        for c in chains_to_align:
            # check for precomputed alignments and maps
            if c in alignments:
                continue
            # otherwise, align and map to PDB resids
            PDBresids = PDBrecord['chain_res'][c]
            seqChain = PDBrecord['chain_seq'][c]
            LOGGER.timeit('_align')
            try:
                a, m = self._quickAlign(seqUniprot, seqChain, PDBresids)
                msg = "Chain {} in {} was quick-aligned".format(c, PDBID)
            except:
                a, m = self._align(seqUniprot, seqChain, PDBresids)
                msg = "Chain {} in {} was aligned".format(c, PDBID)
            LOGGER.report(msg + ' in %.1fs.', '_align')
            # store alignments and maps into PDBmappings
            alignments[c] = a
            maps[c] = m
        return

    def _calcCustomAlignments(self, chains_to_align):
        seqUniprot = self.sequence
        customPDBmapping = self.customPDBmapping
        alignments = customPDBmapping.setdefault('alignments', {})
        maps = customPDBmapping.setdefault('maps', {})
        for c in chains_to_align:
            # check for precomputed alignments and maps
            if c in alignments:
                continue
            # otherwise, align and map to PDB resids
            PDBresids = customPDBmapping['chain_res'][c]
            seqChain = customPDBmapping['chain_seq'][c]
            LOGGER.timeit('_align')
            try:
                a, m = self._quickAlign(seqUniprot, seqChain, PDBresids)
                msg = f"Chain {c} was quick-aligned"
            except:
                LOGGER.info(f"Aligning chain {c} of custom PDB ...")
                a, m = self._align(seqUniprot, seqChain, PDBresids,
                                   print_info=True)
                msg = f"Chain {c} was aligned"
            LOGGER.report(msg + ' in %.1fs.', '_align')
            # store alignments and maps into PDBmappings
            alignments[c] = a
            maps[c] = m
        return

def simpleConversion(SAV_coords):
    # SAVs field: <UniProt ID> <mutation site>
    SAVs = [s.split() for s in SAV_coords]
    SAVs = [f"{s[0]} {s[2]}{s[1]}{s[3]}" for s in SAVs]
    return SAVs

def seqScanning(Uniprot_coord, sequence=None):
    '''Returns a list of SAVs. If the string 'Uniprot_coord' is just a
    Uniprot ID, the list will contain all possible amino acid substitutions
    at all positions in the sequence. If 'Uniprot_coord' also includes a
    specific position, the list will only contain all possible amino acid
    variants at that position. If 'sequence' is 'None' (default), the
    sequence will be downloaded from Uniprot.
    '''
    assert isinstance(Uniprot_coord, str), "Must be a string."
    coord = Uniprot_coord.upper().strip().split()
    assert len(coord) < 3, "Invalid format. Examples: 'Q9BW27' or 'Q9BW27 10'."
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    if sequence is None:
        Uniprot_record = queryUniprot(coord[0])
        sequence = Uniprot_record['sequence   0'].replace("\n", "")
    else:
        assert isinstance(sequence, str), "Must be a string."
        sequence = sequence.upper()
        assert set(sequence).issubset(aa_list), "Invalid list of amino acids."
    if len(coord) == 1:
        # user asks for full-sequence scanning
        positions = range(len(sequence))
    else:
        # user asks for single-site scanning
        site = int(coord[1])
        positions = [site - 1]
        # if user provides only one amino acid as 'sequence', interpret it
        # as the amino acid at the specified position
        if len(sequence) == 1:
            sequence = sequence*site
        else:
            assert len(sequence) >= site, ("Requested position is not found "
                                           "in input sequence.")
    SAV_list = []
    acc = coord[0]
    for i in positions:
        wt_aa = sequence[i]
        for aa in aa_list:
            if aa == wt_aa:
                continue
            s = ' '.join([acc, str(i+1), wt_aa, aa])
            SAV_list.append(s)
    return SAV_list

def mapSAVs2PDB(SAV_coords, custom_PDB=None, refresh=False, **kwargs):
    LOGGER.info('Mapping SAVs to PDB structures...')
    LOGGER.timeit('_mapSAVs2PDB')
    # define a structured array
    PDBmap_dtype = np.dtype([
        ('SAV_coords', 'U25'),
        ('Unique_SAV_coords', 'U25'),
        ('Uniprot_sequence_length', 'i'), 
        ('Asymmetric_PDB_coords', 'U100'),
        ('Asymmetric_PDB_length', 'i'), # Chain length
        ('Asymmetric_PDB_resolved_length', 'i'), # Number of resolved residues
        ('PDB_resolution', 'f'),
        ('BioUnit_PDB_coords', 'U100'),
        ('OPM_PDB_coords', 'U100'),
        ])
    nSAVs = len(SAV_coords)
    mapped_SAVs = np.zeros(nSAVs, dtype=PDBmap_dtype)
    
    groups = {}
    for i, ele in enumerate(SAV_coords):
        acc = ele.split()[0]
        if acc not in groups:
            groups[acc] = {'SAV_coords': [], 'indices': []}
        groups[acc]['SAV_coords'].append(ele)
        groups[acc]['indices'].append(i) # index of each SAV in the original list

    for acc in groups.keys():
        try:
            U2P_map = UniprotMapping(acc, recover_pickle=not(refresh), **kwargs)
            if custom_PDB is not None:
                U2P_map.alignCustomPDB(custom_PDB)
        except Exception as e:
            msg = traceback.format_exc()
            LOGGER.warn(f'Error while mapping {acc}: {msg}')
            # LOGGER.warn(f'Error while mapping {acc}')
            U2P_map = "Cannot map, unable to run " + acc

        if isinstance(U2P_map, str):
            uniq_coords = U2P_map
            for i in groups[acc]['indices']:
                mapped_SAVs[i] = (SAV_coords[i], uniq_coords, 0, 
                                  f'Cannot map, unable to run {acc}', 0, 0, 0, 
                                  None, None)
            continue

        resids, wt_aas, mut_aas = zip(*[ele.split()[1:4] for ele in groups[acc]['SAV_coords']])
        resids = list(map(int, resids))
        indices = groups[acc]['indices']
        if custom_PDB is None:
            r = U2P_map.mapMultipleResidues(resids, wt_aas)
        else:
            r = U2P_map.mapMultipleRes2CustomPDBs(resids, wt_aas)

        for i, (SAV_idx, ele) in enumerate(zip(indices, r)):
            asu_coord, c_seq_len, c_resolved_len, bas_coord, opm_coord, resolution = ele
            uniq_coords = f'{U2P_map.uniq_acc} {resids[i]} {wt_aas[i]} {mut_aas[i]}'
            mapped_SAVs[SAV_idx] = (
                SAV_coords[SAV_idx], 
                uniq_coords, 
                U2P_map.sequence_length, 
                asu_coord, 
                c_seq_len, 
                c_resolved_len, 
                resolution,
                bas_coord, 
                opm_coord
            )

        if isinstance(U2P_map, UniprotMapping):
            U2P_map.savePickle(**kwargs)
    n = sum(mapped_SAVs['Asymmetric_PDB_length'] != 0)
    LOGGER.report(f'{n} out of {nSAVs} SAVs have been mapped to PDB in %.1fs.', '_mapSAVs2PDB')
    return mapped_SAVs

