
import numpy as np
import os
import traceback
from prody import MSA, parseMSA, refineMSA
from prody import calcShannonEntropy, buildMutinfoMatrix
from Bio.Align import substitution_matrices

from ..utils.logger import LOGGER
from .Pfam import run_hmmscan, parse_hmmscan, read_pfam_data, fetchPfamMSA
from .PolyPhen2 import calcPolyPhen2
from .Uniprot import UniprotMapping

__author__ = "Loci Tran"

__all__ = ['SEQfeatures', 'calcSEQfeatures', 'SEQ_FEATS', ]

STD_AA = 'ACDEFGHIKLMNPQRSTVWY'
STD_AA_IDX = {aa: i for i, aa in enumerate(STD_AA)}
AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_IDX = {aa: i for i, aa in enumerate(AA)}

SEQ_FEATS = [
    'entropy', 'ranked_MI', # Pfam
    'wtPSIC', 'deltaPSIC', # PolyPhen-2
    'BLOSUM',
    'phobic_percent', 'delta_phobic_percent',
    'philic_percent', 'delta_philic_percent',
    'charge', 'deltaCharge',
    'polarity', 'deltaPolarity'
]

class SEQfeatures(UniprotMapping):
    def __init__(self, acc, SAV_coords, recover_pickle=False, **kwargs):
        super(SEQfeatures, self).__init__(acc, recover_pickle, **kwargs)
        self.SAV_coords = SAV_coords
        self.resids, self.wt_aas, self.mut_aas = zip(*[ele.split()[1:4] for ele in SAV_coords])
        self.resids = list(map(int, self.resids))
        self.full_SAVs = self.seqScanning()
        self.folder = kwargs.get('folder', '.')

        if "job_directory" in kwargs:
            self.job_directory = kwargs["job_directory"]
        else:
            self.job_directory = '.'
        
    def seqScanning(self):
        """
        Perform sequence scanning for a given Uniprot ID.
        """
        # Perform scanning
        full_SAVs = []
        for i, wt_aa in enumerate(self.sequence):
            for mut_aa in STD_AA:
                if mut_aa == wt_aa:
                    continue
                full_SAVs.append(f'{self.acc} {i+1} {wt_aa} {mut_aa}')
        # self.wt_indices = [AA_IDX[aa] for aa in self.sequence]
        self.wt_indices = [
            STD_AA_IDX[aa] if aa in STD_AA else -1
            for aa in self.wt_aas
        ]
        return full_SAVs
    
    def _searchPfam(self,):
        LOGGER.info('Searching Pfam...')    
        fasta_file = os.path.join(self.job_directory, f'{self.acc}.fasta')
        with open(fasta_file, 'w') as f:
            f.write(f'>{self.acc}\n{self.sequence}\n')
        try: # run hmmscan
            hmmscan_file = run_hmmscan(fasta_file, folder=self.job_directory)
            pfam_data = read_pfam_data()
            Pfam = parse_hmmscan(hmmscan_file, pfam_data)
            self.Pfam = Pfam[self.acc]
        except Exception as e:
            msg = traceback.format_exc()
            LOGGER.warn(msg)
            self.Pfam = str(e)
        os.remove(fasta_file)
        return self.Pfam

    def _sliceMSA(self, msa: MSA):
        acc_name = self.fullRecord['name   0']
        # find sequences in MSA related to the given Uniprot name
        indexes = msa.getIndex(acc_name)
        if indexes is None:
            raise RuntimeError('No sequence found in MSA for {}'.format(acc_name))
        elif type(indexes) is not list:
            indexes = [indexes]
        # slice MSA to include only columns from selected sequences
        cols = np.array([], dtype=int)
        arr = msa._getArray()
        for i in indexes:
            cols = np.append(cols, np.char.isalpha(arr[i]).nonzero()[0])
        cols = np.unique(cols)
        arr = arr.take(cols, 1)
        sliced_msa = MSA(arr, title='refined', labels=msa._labels)
        LOGGER.info('Number of columns in MSA reduced to {}.'.format(sliced_msa.numResidues()))
        return sliced_msa, indexes
    
    def mapUniprot2Pfam(self, PF_ID, msa: MSA, indexes):
        def compareSeqs(s1, s2, tol=0.01):
            if len(s1) != len(s2):
                return None
            seqid = sum(np.array(list(s1)) == np.array(list(s2)))
            seqid = seqid/len(s1)
            if (1 - seqid) > tol:
                return None
            return seqid
        # fetch sequences from Pfam (all locations)
        m = [None]*len(self.sequence)
        sP_list = []
        for i in indexes:
            arr = msa[i].getArray()
            cols = np.char.isalpha(arr).nonzero()[0]
            sP = str(arr[cols], 'utf-8').upper()
            sP_list.append((sP, cols))
        # NB: it's not known which msa index corresponds to each location
        for l in self.Pfam[PF_ID]['locations']:
            r_i = int(l['seq_start']) - 1
            r_f = int(l['seq_end']) - 1
            sU = self.sequence[r_i:r_f+1]
            max_seqid = 0.
            for sP, cols in sP_list:
                seqid = compareSeqs(sU, sP)
                if seqid is None:
                    continue
                if seqid > max_seqid:
                    max_seqid = seqid
                    m[r_i:r_f+1] = cols
                if np.allclose(seqid, 1):
                    break
        # k: Uniprot residue index, v: MSA column index
        return {k: v for k, v in enumerate(m) if v is not None}
    
    def calcEvolNormRank(self, array, i):
        # returns rank in descending order
        order = array.argsort()
        ranks = order.argsort()
        return ranks[i]/len(ranks)

    def calcEvolProperties(self, resid, max_seqs=25000, **kwargs):
        ''' Computes Evol properties, i.e. Shannon entropy, Mutual
        Information and Direct Information, from Pfam Multiple
        Sequence Alignments, for a given residue.
        '''
        # get list of Pfam domains containing resid
        PF_list = [k for k in self.Pfam if any(
            [
                resid >= segment['seq_start'] and resid <= segment['seq_end']
                for segment in self.Pfam[k]['locations']
            ]
        )]
        if len(PF_list) == 0:
            raise RuntimeError(f'No Pfam domain for resid {resid}.')
        if len(PF_list) > 1:
            LOGGER.warn(f'Residue {resid} is found in multiple Pfam domains {PF_list}.')
        # iterate over Pfam families
        for PF in PF_list:
            d = self.Pfam[PF]
            # skip if properties are pre-computed
            if d.get('mapping') is not None:
                continue
            d['mapping'] = None
            d['ref_MSA'] = None
            d['entropy'] = np.nan
            d['MutInfo'] = np.nan
            prefix_PF = PF.split('.')[0]
            try:
                LOGGER.info('Processing {}...'.format(PF))
                # fetch & parse MSA without saving downloaded MSA
                f = fetchPfamMSA(prefix_PF)
                msa = parseMSA(f, **kwargs)
                os.remove(f)
                # slice MSA to match all segments of the Uniprot sequence
                sliced_msa, indexes = self._sliceMSA(msa)
                # get mapping between Uniprot sequence and Pfam domain
                d['mapping'] = self.mapUniprot2Pfam(PF, sliced_msa, indexes)
            except Exception as e:
                LOGGER.warn('{}: {}'.format(PF, e))
                d['mapping'] = str(e)
                continue
            try:
                # refine MSA ('seqid' param. is set as in PolyPhen-2)
                rowocc = 0.6
                while True:
                    sliced_msa = refineMSA(sliced_msa, rowocc=rowocc)
                    rowocc += 0.02
                    if sliced_msa.numSequences() <= max_seqs or rowocc >= 1:
                        break
                ref_msa = refineMSA(sliced_msa, seqid=0.94, **kwargs)
                d['ref_MSA'] = ref_msa
                # compute evolutionary properties
                d['entropy'] = calcShannonEntropy(ref_msa)
                d['MutInfo'] = buildMutinfoMatrix(ref_msa)
            except Exception as e:
                LOGGER.warn('{}: {}'.format(PF, e))
        PF_dict = {k: self.Pfam[k] for k in PF_list}

        # Return entropy and ranked_MI
        entropy = 0.
        rankdMI = 0.
        n = 0
        for ID, Pfam in PF_dict.items():
            if isinstance(Pfam['mapping'], dict):
                n += 1
                idx = Pfam['mapping'][resid - 1]
                entropy += Pfam['entropy'][idx]
                rankdMI += self.calcEvolNormRank(np.sum(Pfam['MutInfo'], axis=0), idx)
        if n == 0:
            raise ValueError("Position couldn't be mapped on any Pfam domain")
        else:
            return entropy/n, rankdMI/n

    def calcPfamfeatures(self):
        features = ['entropy', 'ranked_MI']
        _dtype = np.dtype([(f, 'f') for f in features])
        f = np.full(len(self.resids), np.nan, dtype=_dtype)
        if not self.Pfam:
            pfam = self._searchPfam()
        else:
            pfam = self.Pfam
        if isinstance(pfam, str):
            return f
        for i, resid in enumerate(self.resids):
            try:
                entropy, ranked_MI = self.calcEvolProperties(resid)
                f[i]['entropy'] = entropy
                f[i]['ranked_MI'] = ranked_MI
            except Exception as e:
                msg = traceback.format_exc()
                LOGGER.warn(msg)
        return f
    
    def calcPolyPhen2features(self):
        features = ['wtPSIC', 'deltaPSIC']
        _dtype = np.dtype([(f, 'f') for f in features])
        f = np.full(len(self.resids), np.nan, dtype=_dtype)
        try:
            f = calcPolyPhen2(self.SAV_coords, folder=self.job_directory, filename='_temp_PolyPhen2.txt')
        except Exception as e:
            msg = traceback.format_exc()
            LOGGER.warn(msg)
        return f

        # if all([f in self.feats for f in features]):
        #     if all([isinstance(self.feats[f], np.ndarray) for f in features]):
        #         return
        # try:
        #     # Calculate PolyPhen-2 features for the full list of SAVs
        #     f = calcPolyPhen2(self.full_SAVs) 
        #     # Reshape (n, ) --> (n/19, 19, )
        #     f = f.reshape(-1, 19, )
        #     # Insert a column of zero at the end
        #     f_exd = np.insert(f, 19, 0, axis=1)
        #     # Insert -999 at the corresponding position
        #     for row, idx in enumerate(self.wt_indices):
        #         for col in range(19):
        #             if col >= idx:
        #                 f_exd[row, col+1] = f[row, col]
        #         f_exd[row, idx] = -999
        #     # Extract features
        #     self.feats['wtPSIC'] = f_exd['wtPSIC'][:, 0]
        #     self.feats['deltaPSIC'] = f_exd['deltaPSIC']
        # except Exception as e:
        #     msg = traceback.format_exc()
        #     LOGGER.warn(msg)
        #     for f in features:
        #         self.feats[f] = str(e)

    def calcBLOSUMfeature(self):
        
        blosum62 = substitution_matrices.load("BLOSUM62")
        f = np.zeros(len(self.wt_aas), dtype='f')
        for i, (wt_aa, mut_aa) in enumerate(zip(self.wt_aas, self.mut_aas)):
            f[i] = blosum62[(mut_aa, wt_aa)]
        return f
    
    def calcCHEMfeatures(self):
        features = ['phobic_percent', 'delta_phobic_percent', 
                    'philic_percent', 'delta_philic_percent',
                    'charge', 'deltaCharge', 
                    'polarity', 'deltaPolarity']
        phobic_residues = ['G', 'A', 'V', 'L', 'I', 'M', 'F', 'W', 'P']
        charge_dict = {
            'D': 1, 'E': 1, 'H': 1, 'K': 1, 'R': 1,
            'A': 0, 'C': 0, 'F': 0, 'G': 0, 'I': 0, 'L': 0, 'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
        }
        polarity_dict = {
            'L': 4.9, 'P': 8, 'M': 5.7, 'W': 5.4, 'A': 8.1, 'V': 5.9, 'F': 5.2, 'I': 5.2, 'G': 9, 'S': 9.2, 
            'T': 8.6, 'C': 5.5, 'N': 11.6, 'Q': 10.5, 'Y': 6.2, 'H': 10.4, 'D': 13, 'E': 12.3, 'K': 11.3, 'R': 10.5
        }
        _dtype = np.dtype([(f, 'f') for f in features])
        phobic_count = sum([aa in phobic_residues for aa in self.sequence])
        philic_count = self.sequence_length - phobic_count
        phobic_percent = phobic_count / self.sequence_length * 100
        philic_percent = philic_count / self.sequence_length * 100
        # Calculate features for full_SAVs
        f = np.full(len(self.resids), np.nan, dtype=_dtype)
        f['phobic_percent'] = phobic_percent
        f['philic_percent'] = philic_percent
        for i, (wt_aa, mut_aa) in enumerate(zip(self.wt_aas, self.mut_aas)):
            if (wt_aa in phobic_residues) and (mut_aa in phobic_residues):
                f[i]['delta_phobic_percent'] = 0
                f[i]['delta_philic_percent'] = 0
            elif (wt_aa in phobic_residues) and (mut_aa not in phobic_residues):
                f[i]['delta_phobic_percent'] = (phobic_count - 1) / self.sequence_length * 100 - phobic_percent
                f[i]['delta_philic_percent'] = (philic_count + 1) / self.sequence_length * 100 - philic_percent
            elif (wt_aa not in phobic_residues) and (mut_aa in phobic_residues):
                f[i]['delta_phobic_percent'] = (phobic_count + 1) / self.sequence_length * 100 - phobic_percent
                f[i]['delta_philic_percent'] = (philic_count - 1) / self.sequence_length * 100 - philic_percent
            else:
                f[i]['delta_phobic_percent'] = 0
                f[i]['delta_philic_percent'] = 0
            f[i]['charge'] = charge_dict[wt_aa]
            f[i]['deltaCharge'] = charge_dict[mut_aa] - charge_dict[wt_aa]
            f[i]['polarity'] = polarity_dict[wt_aa]
            f[i]['deltaPolarity'] = polarity_dict[mut_aa] - polarity_dict[wt_aa]
        return f

    def calcFeatures(self, sel_feats: list):
        if not set(sel_feats).issubset(set(SEQ_FEATS)):
            invalid_feats = set(sel_feats) - set(SEQ_FEATS)
            raise ValueError(f'Invalid features: {invalid_feats}')
        # Initialize features
        f_dtype = np.dtype([(f, 'f') for f in sel_feats])
        f = np.full(len(self.resids), np.nan, dtype=f_dtype)
        # Calculate PolyPhen-2 features
        if {'wtPSIC', 'deltaPSIC'}.intersection(set(sel_feats)):
            pph2 = self.calcPolyPhen2features()
            # Extract features
            for feat in ['wtPSIC', 'deltaPSIC']:
                if feat in sel_feats:
                    f[feat] = pph2[feat]
        # Calculate Pfam features
        if {'entropy', 'ranked_MI'}.intersection(set(sel_feats)):
            pfam = self.calcPfamfeatures()
            # Extract features
            for feat in ['entropy', 'ranked_MI']:
                if feat in sel_feats:
                    f[feat] = pfam[feat]
        # Calculate BLOSUM62 features
        if 'BLOSUM' in sel_feats:
            f['BLOSUM'] = self.calcBLOSUMfeature()
        # Calculate CHEM features
        chem_feats = {'phobic_percent', 'delta_phobic_percent', 
            'philic_percent', 'delta_philic_percent',
            'charge', 'deltaCharge', 'polarity', 'deltaPolarity'}
        if chem_feats.intersection(set(sel_feats)):
            chem = self.calcCHEMfeatures()
            # Extract features
            for feat in list(chem_feats):
                if feat in sel_feats:
                    f[feat] = chem[feat]
        return f

def calcSEQfeatures(SAV_coords: list, refresh=False, sel_feats=SEQ_FEATS, 
                    withSAV=False, **kwargs):
    LOGGER.info('Computing sequence features ...')
    LOGGER.timeit('_calcSEQfeatures')
    nSAVs = len(SAV_coords)
    # Convert to numpy array
    SAV_coords = np.array(SAV_coords, dtype=[('SAV_coords', 'U50')])

    # Initialize features
    _dtype = np.dtype([(f, 'f') for f in sel_feats])
    features = np.full(nSAVs, np.nan, dtype=_dtype)
    # Group SAVs by Uniprot ID
    groups = {}
    for i, ele in enumerate(SAV_coords):
        acc = ele['SAV_coords'].split()[0]
        if acc not in groups:
            groups[acc] = []
        groups[acc].append(i)
    ndone = 0
    # Compute features for each group of Uniprot ID
    for acc, indices in groups.items():
        ndone += len(indices)
        try:
            seq = SEQfeatures(acc, SAV_coords[indices]['SAV_coords'], 
                              recover_pickle=not(refresh), **kwargs)
        except Exception as e:
            msg = traceback.format_exc()
            seq = str(e)
            LOGGER.warn(msg)
            continue
        try:
            features[indices] = seq.calcFeatures(sel_feats)
        except Exception as e:
            msg = traceback.format_exc()
            LOGGER.warn(msg)
            for f in sel_feats:
                features[indices][f] = str(e)
        if not isinstance(seq, str):
            seq.savePickle(**kwargs)
        done = ndone / nSAVs
        LOGGER.info(f"SEQ features: {ndone}/{nSAVs} SAVs processed, {acc} [{done:.0%}]")
    LOGGER.report('SEQ features computed in %.2fs.', label='_calcSEQfeatures')
    if withSAV:
        # Concatenate SAV_coords with structured features
        dtype = SAV_coords.dtype.descr + _dtype.descr
        _dtype = np.dtype(dtype)
        _features = np.zeros(nSAVs, dtype=_dtype)
        for f in SAV_coords.dtype.names:
            _features[f] = SAV_coords[f]
        for f in sel_feats:
            _features[f] = features[f]
        return _features
    return features
