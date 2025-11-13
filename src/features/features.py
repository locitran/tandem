import numpy as np
import os

from . import TANDEM_FEATS
from .Uniprot import seqScanning, mapSAVs2PDB
from .PDB import calcPDBfeatures
from .SEQ import calcSEQfeatures
from ..utils.logger import LOGGER 

class Features:

    def __init__(self, query, refresh=False, **kwargs):
        

        # global shape: (nSAVs, )
        # individual shape
        self.model2pred_dtype = np.dtype([
            ('prob', object), # (n_models, ) 
            ('pred', object), # (n_models, )
            ('mode', 'i4'),
            ('decision', 'U20'),
            ('ratio', 'f4'),
            ('path_prob', 'f4'),
            ('path_prob_sem', 'f4'), 
            ('shap', object), # (n_models, n_features)
        ])

        # masked NumPy array that will contain all info about SAVs
        self.data = None
        self.data_dtype = np.dtype([
            # <UniProtID> <mutation site>
            ('SAVs', 'U50'),
            # original Uniprot SAV_coords, extracted from
            # PolyPhen-2's output or imported directly
            ('SAV_coords', 'U50'),
            # Report whether SAV_coords is in training set
            ('is_train', 'U1'),
            # "official" Uniprot SAV identifiers and corresponding
            # PDB coords (if found, otherwise message errors)
            ('Unique_SAV_coords', 'U50'),
            ('Uniprot_sequence_length', 'i4'),
            ('Asymmetric_PDB_coords', 'U100'),
            ('BioUnit_PDB_coords', 'U100'),
            ('OPM_PDB_coords', 'U100'),
            # number of residues in PDB structure (0 if not found)
            ('Asymmetric_PDB_resolved_length', 'i4'),
            # labels for SAVs if available
            ('labels', 'i4'),
            # Predictions from TANDEM and TANDEM transfer learning
            ('tandem', self.model2pred_dtype),
            ('tandem_dimple', self.model2pred_dtype),
        ])

        # number of SAVs
        self.nSAVs = None
        # NumPy array (num_SAVs)x(num_features)
        self.featMatrix = None
        # standardize --> fill nan and standardize
        self.standardize = None
        # classifiers and main feature set
        self.featSet = None
        # custom PDB structure used for PDB features calculation
        self.custom_PDB = None
        # options
        self.options = kwargs
        self.refresh = refresh
        self.saturation_mutagenesis = None
        self.setSAVs(query)
        # map SAVs to PDB structures
        self.Uniprot2PDBmap = None
        self.config = None

    def _isColSet(self, column):
        assert self.data is not None, 'Data array not initialized.'
        return self.data[column].count() != 0

    def setSAVs(self, query):
        assert self.data is None, 'SAV list already set.'
        SAV_dtype = [
            ('acc', 'U10'),
            ('pos', 'i'),
            ('wt_aa', 'U1'),
            ('mut_aa', 'U1')
        ]
        if isinstance(query, str):
            if os.path.isfile(query):
                # 'query' is a filename, with line format 'P17516 135 G E'
                SAVs = np.loadtxt(query, dtype=SAV_dtype)
                SAV_list = ['{} {} {} {}'.format(*s).upper() for s in SAVs]
            elif len(query.split()) < 3:
                # single Uniprot acc (+ pos), e.g. 'P17516' or 'P17516 135'
                SAV_list = seqScanning(query)
                self.saturation_mutagenesis = True
            else:
                # single SAV
                SAV = np.array(query.upper().split(), dtype=SAV_dtype)
                SAV_list = ['{} {} {} {}'.format(*SAV)]
        else:
            # 'query' is a list or tuple of SAV coordinates
            SAVs = np.array([tuple(s.upper().split()) for s in query],
                            dtype=SAV_dtype)
            SAV_list = ['{} {} {} {}'.format(*s) for s in SAVs]
        # store SAV coordinates
        nSAVs = len(SAV_list)
        data = np.ma.masked_all(nSAVs, dtype=self.data_dtype)
        # Assign nan to all columns
        data['SAV_coords'] = SAV_list
        self.data = data
        self.nSAVs = nSAVs

        # SAVs field: <UniProt ID> <mutation site>
        SAVs = [s.split() for s in SAV_list]
        SAVs = [f"{s[0]} {s[2]}{s[1]}{s[3]}" for s in SAVs]
        data['SAVs'] = SAVs
    
    def setLabels(self, labels):
        if labels is None:
            return
        assert self.data is not None, 'SAVs not set.'
        assert len(labels) == self.nSAVs, 'Labels do not match SAVs.'
        assert set(labels).issubset({0, 1}), 'Invalid labels.'
        self.data['labels'] = labels

    def setFeatSet(self, featset):
        assert self.featSet is None, 'Feature set already set.'
        if featset is None:
            featset = TANDEM_FEATS['v1.1']
        elif isinstance(featset, str):
            assert featset in TANDEM_FEATS.keys(), 'Unrecognized feature set.'
            featset = TANDEM_FEATS[featset]
        # check for unrecognized features
        known_feats = TANDEM_FEATS['all']
        for f in featset:
            if f not in known_feats:
                raise RuntimeError(f"Unknown feature: '{f}'")
        if len(set(featset)) != len(featset):
            raise RuntimeError('Duplicate features in feature set.')
        self.featSet = tuple(featset)
        LOGGER.info(f'Selected feature set: {self.featSet}')
        return self.featSet
    
    def setCustomPDB(self, custom_PDB):
        """Set custom PDB structure for PDB features calculation."""
        if custom_PDB is None:
            return
        assert self.custom_PDB is None, 'Custom PDB structure already set.'
        # check if file exists
        self.custom_PDB = custom_PDB
        LOGGER.info(f'Custom PDB structure set to {custom_PDB}')

    def setFeatureMatrix(self, fm):
        assert self.featMatrix is None, 'Feature matrix already set.'
        assert self.featSet is not None, 'Feature set not set.'
        assert self.data is not None, 'SAVs not set.'
        assert len(fm) == self.nSAVs, 'Wrong length.'
        self.featMatrix = fm

    def mapUniprot2PDB(self):
        """Maps each SAV to the corresponding resid in a PDB chain.
        """
        assert self.data is not None, "SAVs not set."
        cols = ['SAV_coords', 'Unique_SAV_coords', 'Asymmetric_PDB_coords', 'Uniprot_sequence_length',
                'BioUnit_PDB_coords', 'OPM_PDB_coords', 'Asymmetric_PDB_resolved_length']
        if not self._isColSet('Asymmetric_PDB_coords'):
            Uniprot2PDBmap = mapSAVs2PDB(
                self.data['SAV_coords'], custom_PDB=self.custom_PDB, 
                refresh=self.refresh, **self.options
            )
            for col in cols:
                self.data[col] = Uniprot2PDBmap[col]
        self.Uniprot2PDBmap = Uniprot2PDBmap

    def getUniprot2PDBmap(self, **kwargs):
        """Maps each SAV to the corresponding resid in a PDB chain.
        """
        if self.Uniprot2PDBmap is None:
            self.mapUniprot2PDB()
        folder = kwargs.get('folder', '.')
        filename = kwargs.get('filename', None)
        os.makedirs(folder, exist_ok=True)
        cols = ['SAV_coords', 'Unique_SAV_coords', 'Asymmetric_PDB_coords', 
                'BioUnit_PDB_coords', 'OPM_PDB_coords', 'Asymmetric_PDB_resolved_length', 'Uniprot_sequence_length']
        # print to file, if requested
        if filename is not None:
            # filename = filename + '-Uniprot2PDB.txt'
            filepath = os.path.join(folder, filename)
            SAVs = self.data['SAVs']
            with open(filepath, 'w') as f:
                f.write(' '.join([
                    f"{'SAV':<15}",
                    f"{'pdbid/chid/resid/aa':<20}",
                    "resolved_len/total_len",
                ]) + '\n')
                for i, s in enumerate(self.data): # type: ignore
                    f.write(' '.join([
                        f"{SAVs[i]:<15}",
                        f"{s['Asymmetric_PDB_coords']:<20}",
                        f"{s['Asymmetric_PDB_resolved_length']}/{s['Uniprot_sequence_length']}",
                    ]) + '\n')
            LOGGER.info(f'Uniprot2PDB map saved to {filepath}')
        return self.Uniprot2PDBmap

    def getSAVs(self, filename=None, folder='.'):
        SAVs = self.data['SAVs']
        if not filename:
            return SAVs
        else:
            filepath = os.path.join(folder, filename)
            with open(filepath, 'w', 1) as f:
                for s in SAVs:
                    f.write(f"{s}\n")
            LOGGER.info(f'SAVs saved to {filename}')
        return filepath
    
    def getFeatMatrix(self, withSAVs=False, withLabels=False, **kwargs):
        """Export feature matrix to a file."""
        if self.featMatrix is None:
            self._calcFeatMatrix()
        folder = kwargs.get('folder', '.')
        filename = kwargs.get('filename', None)
        os.makedirs(folder, exist_ok=True)
        # Concate SAV_coords, labels and features
        sav_coords = np.array(self.data['SAV_coords'])
        labels = np.array(self.data['labels'])
        # Create a new structured array with the desired columns
        dtype = [('SAV_coords', 'U50'), ('labels', 'f')] + \
                [(name, 'f') for name in self.featSet]
        arr = np.zeros(len(sav_coords), dtype=dtype)
        arr['SAV_coords'] = sav_coords
        arr['labels'] = labels
        arr[list(self.featSet)] = self.featMatrix

        if withLabels and withSAVs:
            # Keep all columns
            arr = arr[['SAV_coords', 'labels'] + list(self.featSet)]
        elif withLabels:
            # Remove the SAV_coords column from arr
            arr = arr[['labels'] + list(self.featSet)]
        elif withSAVs:
            # Remove the labels column from arr
            arr = arr[['SAV_coords'] + list(self.featSet)]
            LOGGER.info('SAV_coords column removed from feature matrix.')
        else:
            # Remove the labels and SAV_coords columns from arr
            arr = arr[list(self.featSet)]

        # Save the structured array to a CSV file
        if filename is not None:
            filepath = os.path.join(folder, filename)
            np.savetxt(filepath, arr, delimiter=',', fmt='%s',
                    header=','.join(arr.dtype.names), comments='')
            LOGGER.info(f'Feature matrix saved to {filepath}')
        return arr
    
    def _buildFeatMatrix(self, featset, all_features):
        _dtype = np.dtype([(f, 'f') for f in featset])
        features = np.full(self.nSAVs, np.nan, dtype=_dtype)
        for name in featset:
            # find structured array containing a specific feature
            arrays = [a for a in all_features if name in a.dtype.names]
            if len(arrays) == 0:
                raise RuntimeError(f'Invalid feature name: {name}')
            if len(arrays) > 1:
                LOGGER.warn(f'Multiple values for feature {name}')
            array = arrays[0]
            features[name] = array[name]
            # Report number of missings for each feature
            n_miss = np.sum(np.isnan(array[name]))
            if n_miss > 0:
                LOGGER.warn(f'{n_miss} missing values for feature {name}')
        return features

    def _calcFeatMatrix(self):
        assert self.data is not None, 'SAVs not set.'
        assert self.featSet is not None, 'Feature set not set.'
        # list of structured arrays that will contain all computed features
        all_feats = []
        sel_PDBfeats = TANDEM_FEATS['PDB'].intersection(self.featSet)
        if sel_PDBfeats:
            # compute dynamical features
            f = calcPDBfeatures(self.Uniprot2PDBmap, custom_PDB=self.custom_PDB,
                refresh=self.refresh, sel_feats=sel_PDBfeats, **self.options)
            all_feats.append(f)
        sel_SEQfeats = TANDEM_FEATS['SEQ'].intersection(self.featSet)
        if sel_SEQfeats:
            # compute sequence features
            f = calcSEQfeatures(self.Uniprot2PDBmap['SAV_coords'], 
                refresh=self.refresh, sel_feats=sel_SEQfeats, **self.options)
            all_feats.append(f)
        # build matrix of selected features
        self.featMatrix = self._buildFeatMatrix(self.featSet, all_feats)
