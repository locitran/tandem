
import json, traceback
import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from prody import LOGGER, parsePDB
from prody.measure.contacts import findNeighbors
from Bio.Align import PairwiseAligner

from ..download import get_content, fetchPDB
from ..utils.settings import ROOT_DIR, one2three, RAW_PDB_DIR
from ..utils.timer import getTimer
from ..stand_alone_consurf.main import run

__all__ = ['calcConSurf', 'get_consurf', 'mapIndices']

CONSURFDB_URL = 'https://consurfdb.tau.ac.il/'
pdbDir = ROOT_DIR + '/pdbfile/raw'
consurfDir = ROOT_DIR + '/data/consurf'
dataDir = consurfDir + '/db/2024-10-08'
consurfLookup = consurfDir + '/2024-10-08.json'
customDir = consurfDir + '/db/custom'
# uniref90_2022_05 = consurfDir + '/uniref90_2022_05.fa'
# uniref90_2022_05 = consurfDir + '/uniref90.fasta'
uniref90_2022_05 = consurfDir + '/uniref50.fasta'
os.makedirs(customDir, exist_ok=True)

timer = getTimer('tandem', verbose=True)
with open(consurfLookup) as f:
    consurfLookup = json.load(f)

MATCH_SCORE = 1.0
MISMATCH_SCORE = 0.0
GAP_PENALTY = -1.
GAP_EXT_PENALTY = -0.1

def mapIndices(targetSeq, querySeq):
    """
    Map the indices of the target and query sequences

    Args:
        targetSeq (str): target sequence
        querySeq (str): query sequence
        targetName (str, optional): target name. Defaults to None.
        queryName (str, optional): query name. Defaults to None.

    Returns:
        np.array: target indices
        np.array: query indices

    Gap is represented by -1, and the indices start from 0

    Example:
    >>> mapIndices('ACGT', 'ACGT')
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))
    >>> mapIndices('ACGT', 'CGTA', 'target', 'query')
    (array([ 0,  1,  2,  3, -1]), array([-1,  0,  1,  2,  3]))
    
    target-query.aln
    -----------------
    target indices      0123-
    target            0 ACGT- 4
                      0 -|||- 5
    query             0 -CGTA 4
    query indices       -0123
    """
    aln = _align(targetSeq, querySeq)
    try:
        target_seq, align_dash, query_seq, _ = aln.split('\n')
    except:
        split_aln = aln.split('\n')
        target_seq = ''
        query_seq = ''
        for i, p in enumerate(split_aln):
            p_split = p.split()
            target_seq += p_split[2] if i % 4 == 0 else ''
            query_seq += p_split[2] if i % 4 == 2 else ''

    # Get the indices of the target and query sequences correspondingly
    target_indices = [] ; idx = 0
    for i in range(len(target_seq)):
        if target_seq[i] != '-':
            target_indices.append(idx)
            idx += 1
        else:
            target_indices.append(-1)

    query_indices = [] ; idx = 0
    for i in range(len(query_seq)):
        if query_seq[i] != '-':
            query_indices.append(idx)
            idx += 1
        else:
            query_indices.append(-1)
    return np.array(target_indices), np.array(query_indices)

def _align(target, query):
    """Align two sequences using PairwiseAligner from Bio.Align
    Args:
        target (str): target sequence
        query (str): query sequence
        fileName (str, optional): file name to save the alignment. Defaults to None.
    
    Returns:
        str: alignment in clustal format
    
    Example:
    >>> align2Sequence('ACGT', 'ACGT')
    target            0 ACGT 4
                      0 |||| 4
    query             0 ACGT 4
    """
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = MATCH_SCORE
    aligner.mismatch_score = MISMATCH_SCORE
    aligner.internal_open_gap_score = GAP_PENALTY
    aligner.internal_extend_gap_score = GAP_EXT_PENALTY
    alns = aligner.align(target, query)
    for i, aln in enumerate(alns):
        if i == 1:
            break
    return aln.format()

def _parse(unique_chain):
    url = f'{CONSURFDB_URL}DB/{unique_chain}/{unique_chain}_consurf_summary.txt'
    # url = f'{CONSURFDB_URL}DB_NEW/{pdbid}/{unique_chain}/{pdbid}_{unique_chain}_consurf_grades.txt'
    content = get_content(url)
    if content is None:
        return None

    try:
        lines = content.split('\n')
        lines = [line.strip() for line in lines[:-5]]
        for i, line in enumerate(lines):
            if line == '(normalized)':
                cols = lines[i-1]
                lines = lines[i+1:]
                break
        cols = [col.strip() for col in cols.split('\t') if col != '']

        data = []
        for line in lines:
            line = line.split('\t')
            line = [txt.strip() for txt in line if txt.strip() != '']
            if len(line) == 9 and line[0].isdigit():
                if line[2] != '-':
                    line[2] = line[2].split(':')[0]
            data.append(line)
        df = pd.DataFrame(data, columns=cols)
        return df
    except Exception as e:
        LOGGER.warning(f'Error parsing {unique_chain}, {e}')
        return None

def get_consurf(unique_chain, folder='.') -> pd.DataFrame:
    """Run the Consurf database for a protein.

    Returns:
    --------
    df: pd.DataFrame, the conservation data for the protein
    """

    outpath = os.path.join(folder, f'{unique_chain}.tsv')
    if os.path.exists(outpath):
        return pd.read_csv(outpath, sep='\t')
    else:
        df = _parse(unique_chain)
        if df is None:
            return None
        df.to_csv(outpath, sep='\t', index=False)
        return df

def getConSurffile(pdb, chid, folder='.'):
    """Get the consurf file for a given PDB ID and chain ID.
    pdb: PDB ID or PDB file
        - pdb: PDB ID 
            1. Check consurfLookup for PDB ID + chid
            2. Run stand_alone_consurf if not found
        
        - pdb: PDB file
            1. Check existence of the file f'{pdb}_{chid}.tsv' in customDir
            2. Run stand_alone_consurf for given chain if not found

    folder provided in case of running stand_alone_consurf
    Returns:
        df: DataFrame of output consurffile

    Example:        
        from src.features.consurf import getConSurffile
        pdb = '4xr8'
        pdb = '/home/newloci/tandem/src/stand_alone_consurf/pkd1/fold_1xpkd1_model_0_A.pdb'
        chid = 'A'
        consurf = getConSurffile(pdb, chid)
    """
    if not os.path.isfile(pdb):
        pdbID = pdb.upper()
        if pdbID in consurfLookup:
            if chid in consurfLookup[pdbID]:
                uniqueChain = consurfLookup[pdbID][chid]
                consurffile = os.path.join(dataDir, f'{uniqueChain}.tsv')
                return pd.read_csv(consurffile, sep='\t')
        else:
            pdb = fetchPDB(pdbID, format='pdb', compressed=False, folder=RAW_PDB_DIR)
            if pdb is None:
                raise ValueError(f'Cannot download {pdbID}')
    else:
        pdbID = os.path.basename(pdb).split('.')[0]

    # Search uniqueChain in customDir
    uniqueChain = f'{pdbID}_{chid}'
    consurffile = os.path.join(customDir, f'{uniqueChain}.tsv')
    if os.path.isfile(consurffile):
        return pd.read_csv(consurffile, sep='\t')
    
    LOGGER.info(f'Running consurf for {pdbID} {chid}')
    # If not found, run stand_alone_consurf
    out = run(
        query=pdbID,
        structure=pdb,
        chain=chid,
        DB=uniref90_2022_05,
        work_dir=folder,
        algorithm="HMMER"
    )
    # Parse the consurf file
    data = []
    with open(out, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip() != '']
        for i, line in enumerate(lines):
            if line.startswith('POS'):
                cols = lines[i]
                lines = lines[i+1:]
                break
        cols = [col.strip() for col in cols.split('\t') if col != '']
        for line in lines:
            line = line.split('\t')
            if len(line) == 10 and line[0].isdigit():
                data.append(line)
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(consurffile, sep='\t', index=False)
    return df

def calcConSurf(pdb, chids, resids, wt_aas, folder='.'):
    """Retrieve consurf and ACNR scores for the target and contact residues
    folder is used to store the file of consurf relatives in case of running stand_alone_consurf

    target (tgt), contact (cnt)
    """
    assert len(chids) == len(resids) == len(wt_aas), 'chids, resids, and wt_aas must have the same length'
    # _dtype = np.dtype([('consurf', 'f'), 
    #                    ('ACNR', 'f')]
    #                    )
    _dtype = np.dtype([
        ('consurf', 'f4'), 
        ('ACNR', 'f4'),
        ('consurf_color', 'i4'),
    ])
    features = np.full(len(chids), np.nan, dtype=_dtype)
    
    # Read the PDB file
    # custom or alphafold
    if os.path.isfile(pdb):
        pdbID = pdb
        pdb = parsePDB(pdb, model=1)
    else: # pdbID 
        pdbID = pdb
        pdbpath = fetchPDB(pdbID, format='pdb', folder=RAW_PDB_DIR)
        if pdbpath is not None:
            pdb = parsePDB(pdbpath, model=1)
        else:
            raise ValueError(f'Cannot download {pdbID}')
        
    pdb_chids = set(pdb.ca.getChids())
    ca = pdb.protein.ca
    LOGGER.timeit('_calcConSurf')
    for i, (tgt_chid, tgt_resid, wt_aa) in enumerate(zip(chids, resids, wt_aas)):
        if tgt_chid not in pdb_chids:
            LOGGER.warn(f'{tgt_chid} not in {pdbID}')
            continue
        # LOGGER.info(f"Processing {pdbID} {tgt_uniqueChain} {wt_aa}")
        tgt_chain = ca.select(f'chain {tgt_chid}').copy()
        target = tgt_chain.select(f'resnum `{tgt_resid}` and resname {one2three[wt_aa]}')
        if target is None:
            LOGGER.warn(f'{pdbID} {tgt_chid} {tgt_resid} not found')
            continue
        try: # Get consurf file for the target
            df_tgt = getConSurffile(pdbID, tgt_chid, folder=folder)
        except Exception as e: 
            msg = traceback.format_exc()
            LOGGER.warn(f'Error processing {pdbID} {tgt_chid}: {msg}')
            continue
    
        # LOGGER.info(f"Processing {pdbID} {tgt_chid} {tgt_resid}")
        contacts = findNeighbors(target, 7.3, ca)
        contacts = [contact for (target, contact, distance) in contacts if distance != 0]
        tgt_consurf_seq = df_tgt.SEQ.to_string(index=False).replace('\n', '').replace(' ', '')
        tgt_seq = tgt_chain.getSequence()
        # Map the indices
        tgt_indices, tgt_consurf_indices = mapIndices(tgt_seq, tgt_consurf_seq)
        tgt_resindex = target.getResindices()[0]
        
        # Retrieve consurf scores for target
        tgt_idx = np.where(tgt_indices == tgt_resindex)[0][0]
        tgt_idx = tgt_consurf_indices[tgt_idx]
        tgt_score = float(df_tgt.loc[tgt_idx]['SCORE'])
        tgt_color = df_tgt.loc[tgt_idx]['COLOR'] # some cases color contains "*" -> remove it

        if isinstance(tgt_color, str) and '*' in tgt_color:
            tgt_color = int(tgt_color.split('*')[0]) + 10
        else:
            tgt_color = int(tgt_color)

        features['consurf'][i] = tgt_score
        features['consurf_color'][i] = tgt_color
        # LOGGER.info(f'Target {target}, {tgt_chid}, {tgt_resid}, {tgt_resindex},  {tgt_score}')
        # Retrieve consurf scores for contacts
        cnt_scores = []
        for contact in contacts:
            cnt_chid = contact.getChid()
            cnt_resnum = contact.getResnum()
            cnt_resicode = contact.getIcode()
            cnt_resname = contact.getResname()
            # Reset residue indices
            cnt_chain = ca.select(f'chain {cnt_chid}').copy()
            cnt_reindex = cnt_chain.select(f'resnum `{cnt_resnum}` and resname {cnt_resname}')
            if cnt_reindex is None:
                LOGGER.warn(f'{pdbID} {cnt_chid} {cnt_resnum} not found')
                continue
            cnt_resindex = cnt_reindex.getResindices()[0]
            # LOGGER.info(f'Contact {contact}, {cnt_resnum}, {contact.getResname()} {cnt_resindex}, {cnt_chid}')
            if cnt_chid == tgt_chid:
                cnt_idx = np.where(tgt_indices == cnt_resindex)[0][0]
                cnt_idx = tgt_consurf_indices[cnt_idx]
                s = float(df_tgt.loc[cnt_idx]['SCORE'])
                cnt_scores.append(s)
            else:
                cnt_seq = cnt_chain.getSequence()
                if len(cnt_seq) <= 35:
                    LOGGER.warn(f'{pdbID} {cnt_chid} too short no consurf {len(cnt_seq)}')
                    continue
                try:
                    df_cnt = getConSurffile(pdbID, cnt_chid, folder=folder)
                except Exception as e:
                    LOGGER.warn(f'Error process Contact {pdbID} {cnt_chid}: {str(e)}')
                    continue
                cnt_consurf_seq = df_cnt.SEQ.to_string(index=False).replace('\n', '').replace(' ', '')
                cnt_indices, cnt_consurf_indices = mapIndices(cnt_seq, cnt_consurf_seq)
                # LOGGER.info(f'Contact {cnt_indices}, {cnt_consurf_indices}')
                # Retrieve consurf scores for contact
                cnt_idx = np.where(cnt_indices == cnt_resindex)[0][0]
                cnt_idx = cnt_consurf_indices[cnt_idx]
                s = float(df_cnt.loc[cnt_idx]['SCORE'])
                cnt_scores.append(s)
                # LOGGER.info(f'Contact {contact}, {cnt_resnum}, {contact.getResname()} {cnt_resindex}, {cnt_chid} {s}')
        features['ACNR'][i] = np.nanmean(cnt_scores)
    LOGGER.report('ConSurf features calculated in %.2fs.', label='_calcConSurf')
    return features
