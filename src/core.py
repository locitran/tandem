import numpy as np
import os
import traceback

from .utils.logger import LOGGER
from .features import Uniprot, PDB, SEQ, TANDEM_FEATS
from .utils.settings import ROOT_DIR
from .features.PolyPhen2 import printSAVlist

__all__ = ['Tandem']
__version__ = '0.1.0'
__author__ = 'Loci Tran'

class Tandem:

    def __init__(self, query, refresh=False, **kwargs):
        
        # masked NumPy array that will contain all info about SAVs
        self.data = None
        self.data_dtype = np.dtype([
            # original Uniprot SAV_coords, extracted from
            # PolyPhen-2's output or imported directly
            ('SAV_coords', 'U50'),
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
            ('labels', 'f4'),
        ])
        # number of SAVs
        self.nSAVs = None
        # NumPy array (num_SAVs)x(num_features)
        self.featMatrix = None
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
                SAV_list = Uniprot.seqScanning(query)
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
    
    def setLabels(self, labels):
        assert self.data is not None, 'SAVs not set.'
        assert len(labels) == self.nSAVs, 'Labels do not match SAVs.'
        if len(labels) != self.nSAVs:
            LOGGER.warn(f'Number of labels ({len(labels)}) does not match number of SAVs ({self.nSAVs}).')
        else:
            self.data['labels'] = labels

    def setFeatSet(self, featset):
        assert self.featSet is None, 'Feature set already set.'
        if isinstance(featset, str):
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
        """Set custom PDB structure for PDB features calculation.
        """
        assert self.custom_PDB is None, 'Custom PDB structure already set.'
        # check if file exists
        self.custom_PDB = custom_PDB
        LOGGER.info(f'Custom PDB structure set to {custom_PDB}')

    def mapUniprot2PDB(self):
        """Maps each SAV to the corresponding resid in a PDB chain.
        """
        assert self.data is not None, "SAVs not set."
        cols = ['SAV_coords', 'Unique_SAV_coords', 'Asymmetric_PDB_coords', 'Uniprot_sequence_length',
                'BioUnit_PDB_coords', 'OPM_PDB_coords', 'Asymmetric_PDB_resolved_length']
        if not self._isColSet('Asymmetric_PDB_coords'):
            Uniprot2PDBmap = Uniprot.mapSAVs2PDB(
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
            filename = filename + '-Uniprot2PDB.txt'
            filepath = os.path.join(folder, filename)
            with open(filepath, 'w') as f:
                f.write(' '.join([
                    f'{cols[0]:<18}', f'{cols[1]:<18}', f'{cols[2]:<35}', 
                    f'{cols[3]:<35}', f'{cols[4]:<35}', f'{cols[5]:<35}', f'{cols[6]:<35}']) + '\n')
                for s in self.data: # type: ignore
                    f.write(' '.join([
                        f'{s[cols[0]]:<18}', f'{s[cols[1]]:<18}', f'{s[cols[2]]:<35}',
                        f'{s[cols[3]]:<35}', f'{s[cols[4]]:<35}', f'{s[cols[5]]:<35}', f'{s[cols[6]]:<35}']) + '\n')
            LOGGER.info(f'Uniprot2PDB map saved to {filepath}')
        return self.Uniprot2PDBmap
    
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
            filepath = os.path.join(folder, filename + '-features.csv')
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
            f = PDB.calcPDBfeatures(self.Uniprot2PDBmap, custom_PDB=self.custom_PDB,
                refresh=self.refresh, sel_feats=sel_PDBfeats, **self.options)
            all_feats.append(f)
        sel_SEQfeats = TANDEM_FEATS['SEQ'].intersection(self.featSet)
        if sel_SEQfeats:
            # compute sequence features
            f = SEQ.calcSEQfeatures(self.Uniprot2PDBmap['SAV_coords'], 
                refresh=self.refresh, sel_feats=sel_SEQfeats, **self.options)
            all_feats.append(f)
        # build matrix of selected features
        self.featMatrix = self._buildFeatMatrix(self.featSet, all_feats)

    # Calculate predictions

    def predictSAVs(self, models=None, r20000=None, 
                    model_names='TANDEM-DIMPE_v1', **kwargs):
        assert self.featMatrix is not None, 'Feature matrix not set.'
        # Convert the feature matrix to a NumPy array
        fm = self.featMatrix.view(np.float32).reshape(self.nSAVs, -1)
        try:
            LOGGER.timeit("_predict")
            from .predict.inference import ModelInference
            mi = ModelInference(folder=models, r20000=r20000, featSet=list(self.featSet))
            mi.calcPredictions(fm)
            LOGGER.report("Predictions computed in %.2fs.", "_predict")
            
        except:
            msg = traceback.format_exc()
            LOGGER.error(f'Error in prediction: {msg}')

        """Export predictions to a file."""

        folder = kwargs.get('folder', '.')
        filename = kwargs.get('filename', None)
        os.makedirs(folder, exist_ok=True)

        # Retrieve probs, votes, decisions from mi
        probs = mi.probs
        votes = mi.votes
        decisions = mi.decisions
        final_probs = mi.final_probs

        if filename is not None:
            cols = ['SAVs', 'Probability', 'Decision', 'Voting']
            report = filename + '-report.txt'
            filepath = os.path.join(folder, report)
            with open(filepath, 'w') as f:
                # Write the header
                for col in cols:
                    f.write(f'{col:<18}')
                f.write('\n')
                # Write the data
                for i in range(self.nSAVs):
                    decision = 'Pathogenic' if decisions[i] == 1 else 'Benign'
                    f.write(f'{self.data["SAV_coords"][i]:<18}')
                    f.write(f'{final_probs[i].item():<18.4f}')
                    f.write(f'{decision:<18s}')
                    f.write(f'{votes[i].item()*100:<18.1f}\n')
            LOGGER.info(f'Report saved to {filepath}')

            cols = ['SAVs']
            cols += [f'{model_names[:6]}_{i}' for i in range(len(mi.models))]
            full_preds = filename + '-full_predictions.txt'
            filepath = os.path.join(folder, full_preds)
            with open(filepath, 'w') as f:
                # Write the header
                for col in cols:
                    f.write(f'{col:<18}')
                f.write('\n')
                # Write the data
                for i in range(self.nSAVs):
                    f.write(f'{self.data["SAV_coords"][i]:<18}')
                    for j in range(len(mi.models)):
                        f.write(f'{probs[i][j]:<18.4f}')
                    f.write('\n')
            LOGGER.info(f'Predictions saved to {filepath}')
        return probs, votes, decisions, final_probs


### Functions

def calcFeatures(query, labels=None, job_name='tandem-dimple', custom_PDB=None,
                 refresh=False, featSet='v1.1', 
                 withSAVs=False, withLabels=False, **kwargs):
    """Main function to calculate features for SAVs."""
    # Create a directory for the job
    job_directory = os.path.join(ROOT_DIR, 'jobs', job_name)
    os.makedirs(job_directory, exist_ok=True)
    ## LOGGER
    logfile = os.path.join(job_directory, 'log.txt')
    LOGGER.start(logfile)
    # Set up the Tandem object
    kwargs['job_directory'] = job_directory
    if 'folder' not in kwargs:
        kwargs['folder'] = os.path.join(ROOT_DIR, 'data') 
    os.makedirs(kwargs['folder'], exist_ok=True)
    t = Tandem(query, refresh=refresh, **kwargs)

    # Save SAVs to a file
    printSAVlist(t.data['SAV_coords'], os.path.join(job_directory, 'SAVs.txt'))

    # Set custom PDB structure
    if custom_PDB:
        t.setCustomPDB(custom_PDB)
    
    # Set labels
    if labels is not None:
        t.setLabels(labels)
    
    # Save the Uniprot2PDB map
    t.getUniprot2PDBmap(folder=job_directory, filename=job_name)

    # Set up the feature set
    featSet = t.setFeatSet(featSet)

    return t.getFeatMatrix(
        withSAVs=withSAVs, withLabels=withLabels, 
        folder=job_directory, filename=job_name
    )

def predictSAVs(query, job_name='test', custom_PDB=None, 
                  featSet='ordered_final_v1', refresh=False, 
                  models=None, model_names="TANDEM-DIMPE_v1",
                  r20000=None,
                  **kwargs):
    """Main function to predict the effect of SAVs on protein stability.

    Parameters
    ----------
    query : str or list
        A list of SAV coordinates
        E.g. ['P17516 135 G E', 'P17516 136 G E']
    
    custom_PDB : str, optional
        Path to custom PDB file. If not provided, the default PDB file will be used.
    
    featset : str or list, optional
        Feature set to use. If not provided, the default feature set will be used.
        Available feature sets: 'ordered_final_v1'
    
    refresh : bool, optional
        If True, refresh the cached data, recompute features and save them to disk.
        If False, use the cached data.

    kwargs : dict, optional
        folder : str
            Folder to save the feature matrix.
            

    """
    # Create a directory for the job
    job_directory = os.path.join(ROOT_DIR, 'jobs', job_name)
    os.makedirs(job_directory, exist_ok=True)
    ## LOGGER
    logfile = os.path.join(job_directory, 'log.txt')
    LOGGER.start(logfile)

    # Set up the Tandem object
    tandem = Tandem(query, custom_PDB=custom_PDB, refresh=refresh, **kwargs)
    ## Save the Uniprot2PDB map
    tandem.getUniprot2PDBmap(folder=job_directory, filename='tandem')
    # Set up the feature set
    featSet = tandem.setFeatSet(featSet)
    # Compute the feature matrix
    tandem._calcFeatMatrix(refresh=refresh, **kwargs)
    # Save the feature matrix
    tandem.getFeatMatrix(folder=job_directory, filename='tandem')
    # test_fm = tandem.featMatrix

    # Save the SAVs to a file
    printSAVlist(tandem.data['SAV_coords'], os.path.join(job_directory, 'SAVs.txt'))
    # Save the feature matrix