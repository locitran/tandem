import numpy as np
import pickle
import datetime
import os
import traceback
import pandas as pd
import ronn
from collections import defaultdict

from prody import parsePDB, writePDB
from prody import calcPerturbResponse, calcMechStiff, sliceModel
from prody.atomic import sliceAtoms

from ..utils.logger import LOGGER
from ..dynamics.ENM import GNM, envGNM, ANM, envANM
from ..dynamics.entropy import calcSpectralEntropy
from ..dynamics.paa import calcShapeFactors
from ..dynamics.contact import calcAG, calcDisulfideBonds
from ..dynamics.contact import calcRGandDcom, calcBJCEnergy, calcLside
from ..utils.settings import FIX_PDB_DIR
from ..fixer import createMutationfile, fixPDB
from .dssp import calcSecondary
from .naccess import calcAccessibility
from .consurf import calcConSurf
from .hbplus import calcHbond
from .PropKa import calcChargepH7
from .Uniprot import verifyAF

MAX_NUM_RESIDUES = 21000
"""Hard-coded maximum size of PDB structures that can be handled by the
class :class:`PDBfeatures()`."""

__all__ = ['PDBfeatures', 'PDB_FEATS', 'STR_FEATS', 'DYN_FEATS']

STR_FEATS = ['IDRs', 'SSbond', 'SF1', 'SF2', 'SF3', 'chain_length', 'protein_length',
             'AG1', 'AG3', 'AG5', 'ACR', 'Rg', 'Dcom',
             'loop_percent', 'helix_percent', 'sheet_percent', 'dssp',
             'SASA', 'SASA_in_complex', 'deltaSASA', 'Hbond', 'charge_pH7',
             'wtBJCE', 'mutBJCE', 'deltaBJCE', 'Lside', 'deltaLside',
             'DELTA_Rg', 'DELTA_Dcom', 'DELTA_SASA', 'DELTA_ACR', 
             'DELTA_Hbond', 'DELTA_DSS', 'DELTA_charge_pH7',
             'consurf', 'ACNR', 'consurf_color']
# 'consurf', 'ACNR' are are supposed to be SEQ features, but
# they are calculated here for convenience

DYN_FEATS = ['GNM_Ventropy', 'GNM_rmsf_overall',
             'GNM_Eigval1', 'GNM_Eigval2', 'GNM_Eigval5_1',
             'GNM_SEall', 'GNM_SE20',
             'GNM_V1', 'GNM_rankV1',
             'GNM_V2', 'GNM_rankV2',
             'GNM_co_rank', 'GNM_displacement',
             'GNM_MC1', 'GNM_MC2',

             'ANM_effectiveness', 
             'ANM_sensitivity',
             'ANM_stiffness']
"""List of available dynamical features."""

PDB_FEATS = STR_FEATS + [f + e for f in DYN_FEATS
                         for e in ['_chain', '_reduced', '_sliced', '_full']]

class PDBfeatures:

    def __init__(self, pdbPath: str, format='asu', n_modes='all', recover_pickle=False, **kwargs):
        assert os.path.isfile(pdbPath), f'File not found: {pdbPath}'
        assert type(recover_pickle) is bool
        # definition and initialization of variables
        self.pdbPath = os.path.abspath(pdbPath)
        self.folder = os.path.dirname(pdbPath)
        self._pdb = None
        
        basename = os.path.basename(pdbPath).split('.')[0]
        if format == 'custom' or format == 'af':
            self.pdbID = basename
        else:
            self.pdbID = basename[:4]
        
        self.format = format
        self.n_modes = n_modes
        self.chids = None
        self.resids = None
        self.feats = None
        self._gnm = None
        self._anm = None
        self.timestamp = None

        if "job_directory" in kwargs:
            self.job_directory = kwargs["job_directory"]
        else:
            self.job_directory = '.'

        if recover_pickle:
            try:
                self.recoverPickle(**kwargs)
            except Exception:
                LOGGER.warning(f'Unable to recover pickle: PDBfeatures-{self.pdbID}-{self.format}.pkl')
                self.refresh()
        else:
            self.refresh()
        return

    def getPDB(self):
        """Returns the parsed PDB structure as an :class:`AtomGroup` object."""
        if self._pdb is None:
            self._pdb = parsePDB(self.pdbPath, model=1)
        return self._pdb

    def refresh(self):
        """Deletes all precomputed ENM models and features, and resets
        time stamp."""
        pdb = self.getPDB()
        ca = pdb.ca.copy()
        self.chids = set(ca.getChids())
        self.resids = {chID: ca[chID].getResnums()
                       for chID in self.chids}
        self._gnm = {}
        self._anm = {}
        for env in ['chain', 'reduced', 'sliced', 'full']:
            self._gnm[env] = {chID: None for chID in self.chids}
            self._anm[env] = {chID: None for chID in self.chids}
        self.feats = {chID: {} for chID in self.chids}
        self.timestamp = str(datetime.datetime.utcnow())
        return

    def recoverPickle(self, days=30*6, **kwargs):
        """Looks for precomputed pickle for the current PDB structure.

        :arg folder: path of folder where pickles are stored. If not specified,
            pickles will be searched for in the local Rhapsody installation
            folder.
        :type folder: str
        :arg filename: name of the pickle. If not specified, the default
            filename ``'PDBfeatures-[PDBID].pkl'`` will be used. If a PDBID is
            not found, user must specify a valid filename.
        :type filename: str
        :arg days: number of days after which a pickle will be considered too
            old and won't be recovered.
        :type days: int
        """
        filename = kwargs.get('filename', self.pdbID)
        folder = kwargs.get('folder', '.')
        folder = os.path.join(folder, 'pickles/pdb')
        filename = f'PDBfeatures-{self.pdbID}-{self.format}.pkl'
        pickle_path = os.path.join(folder, filename)
        if not os.path.isfile(pickle_path):
            raise IOError("File '{}' not found".format(filename))
        recovered_self = pickle.load(open(pickle_path, "rb"))
        
        # check consistency of recovered data
        if self.pdbID != recovered_self.pdbID:
            raise ValueError('Name in recovered pickle ({}) does not match.'
                             .format(recovered_self.pdbID))
        if self.n_modes != recovered_self.n_modes:
            raise ValueError('Num. of modes in recovered pickle ({}) does not match.'
                             .format(recovered_self.n_modes))
        # check timestamp and ignore pickles that are too old
        date_format = "%Y-%m-%d %H:%M:%S.%f"
        t_old = datetime.datetime.strptime(
            recovered_self.timestamp, date_format)
        t_now = datetime.datetime.utcnow()
        Delta_t = datetime.timedelta(days=days)
        if t_old + Delta_t < t_now:
            raise RuntimeError('Pickle was too old and was ignored.')
        # import recovered data
        self.chids = recovered_self.chids
        self.resids = recovered_self.resids
        self.feats = recovered_self.feats
        self._gnm = recovered_self._gnm
        self._anm = recovered_self._anm
        self.timestamp = recovered_self.timestamp
        LOGGER.info("Pickle '{}' recovered.".format(filename))
        return

    def savePickle(self, **kwargs):
        """Stores a pickle of the current class instance. The pickle will
        contain all information and precomputed features, but not GNM and ANM
        models. In case a PDBID is missing, the parsed PDB :class:`AtomGroup`
        is stored as well.

        :arg folder: path of the folder where the pickle will be saved. If not
            specified, the local Rhapsody installation folder will be used.
        :type folder: str
        :arg filename: name of the pickle. By default, the pickle will be
            saved as ``'PDBfeatures-[PDBID].pkl'``. If a PDBID is not defined,
            the user must provide a filename.
        :type filename: str
        :return: pickle path
        :rtype: str
        """
        filename = kwargs.get('filename', self.pdbID)
        folder = kwargs.get('folder', '.')
        folder = os.path.join(folder, 'pickles/pdb')
        os.makedirs(folder, exist_ok=True)
        filename = f'PDBfeatures-{self.pdbID}-{self.format}.pkl'
        pickle_path = os.path.join(folder, filename)
        # do not store GNM and ANM instances.
        # If a valid PDBID is present, do not store parsed PDB
        # as well, since it can be easily fetched again
        cache = (self._pdb, self._gnm, self._anm)
        if self.pdbID is not None:
            self._pdb = None
        self._gnm = {}
        self._anm = {}
        for env in ['chain', 'reduced', 'sliced', 'full']:
            self._gnm[env] = {chID: None for chID in self.chids}
            self._anm[env] = {chID: None for chID in self.chids}
        # write pickle
        pickle.dump(self, open(pickle_path, "wb"))
        # restore temporarily cached data
        self._pdb, self._gnm, self._anm = cache
        LOGGER.info("Pickle '{}' saved.".format(filename))
        return pickle_path

    def _checkNumCalphas(self, ag):
        n_ca = ag.ca.numAtoms()
        if n_ca > MAX_NUM_RESIDUES:
            m = f'Too many C-alphas: {n_ca}. Max. allowed: {MAX_NUM_RESIDUES}'
            raise RuntimeError(m)

    #####################################
    # ENM features
    """
    env:
    - 'chain': regular GNM/ANM model on the selected chain
    - 'reduced': envGNM/envANM model on the selected chain, defined as system, and 
        environment, defined as other chains, membrane NE1 atoms and nucleic.
    - 'sliced': regular GNM/ANM model on the whole structure, but sliced to the selected
        chain. 
    - 'full': regular GNM model on the whole structure, no slicing, in case of only protein.
        envGNM model on the whole structure, in case of protein+membrane/+nucleic.
    """

    def calcGNM(self, chID, env='chain'):
        """Builds GNM model for the selected chain.

        :arg chID: chain identifier
        :type chID: str
        :arg env: environment model, i.e. ``'chain'``, ``'reduced'`` or
            ``'sliced'``
        :type env: str
        :return: GNM model
        :rtype: :class:`GNM`
        """
        assert env in ['chain', 'reduced', 'sliced', 'full']
        gnm_e = self._gnm[env]
        n = self.n_modes
        if gnm_e[chID] is None:
            pdb = self.getPDB()
            if env == 'full':
                if self.format != 'af':
                    system = pdb.ca
                    self._checkNumCalphas(system)
                    selstr = '(resname NE1 name Q1) or ' # membrane NE1 atoms
                    selstr += '(nucleic name P C4\' C2)' # nucleic atoms
                    environment = pdb.select(selstr)
                    if environment is None:
                        gnm = GNM()
                        LOGGER.info('Building GNM model on the whole structure')
                        LOGGER.info('No environment, using GNM')
                        gnm.buildKirchhoff(system, cutoff=7.3, gamma=1.0)
                        gnm.calcModes(n_modes=n)
                    else:
                        gnm = envGNM()
                        gnm.buildHessian(system, environment, cutoff=7.3, gamma=1.0)
                        gnm.calcModes(n_modes=n)
                    for c in self.chids:
                        gnm_e[c] = gnm
                # af format, system is low/high/very high beta (confidence)
                # and environment is very low beta (confidence)
                else:
                    system = pdb.ca.select('beta >= 50')
                    environment = pdb.ca.select('beta < 50')
                    if environment is None:
                        gnm = GNM()
                        LOGGER.info('Building GNM model on the whole structure')
                        LOGGER.info('No environment, using GNM')
                        gnm.buildKirchhoff(system, cutoff=7.3, gamma=1.0)
                        gnm.calcModes(n_modes=n)
                    else:
                        gnm = envGNM()
                        gnm.buildHessian(system, environment, cutoff=7.3, gamma=1.0)
                        gnm.calcModes(n_modes=n)
                    for c in self.chids:
                        gnm_e[c] = gnm
            elif env == 'chain':
                ca = pdb.ca[chID]
                self._checkNumCalphas(ca)
                gnm = GNM()
                LOGGER.info('Building GNM model on the whole structure')
                LOGGER.info('No environment, using GNM')
                gnm.buildKirchhoff(ca, cutoff=7.3, gamma=1.0)
                gnm.calcModes(n_modes=n)
                gnm_e[chID] = gnm
            elif env == 'reduced':
                system = pdb.ca[chID]
                self._checkNumCalphas(system)
                selstr = f'(not chain `{chID}` and name CA) or '
                selstr += '(resname NE1 name Q1) or ' # membrane NE1 atoms
                selstr += '(nucleic name P C4\' C2)' # nucleic atoms
                environment = pdb.select(selstr)
                if environment is None:
                    gnm = GNM()
                    gnm.buildKirchhoff(system, cutoff=7.3, gamma=1.0)
                    gnm.calcModes(n_modes=n)
                    gnm_e[chID] = gnm
                else:
                    gnm = envGNM()
                    gnm.buildHessian(system, environment, cutoff=7.3, gamma=1.0)
                    gnm.calcModes(n_modes=n)
                    gnm_e[chID] = gnm
            else: # env == 'sliced'
                ca = pdb.ca
                self._checkNumCalphas(ca)
                gnm_full = GNM()
                gnm_full.buildKirchhoff(ca, cutoff=7.3, gamma=1.0)
                gnm_full.calcModes(n_modes=n)
                for c in self.chids:
                    sel = f'chain `{c}`'
                    gnm, _ = sliceModel(gnm_full, ca, sel)
                    gnm_e[c] = gnm
        return self._gnm[env][chID]

    def calcANM(self, chID, env='chain'):
        """Builds ANM model for the selected chain.

        :arg chID: chain identifier
        :type chID: str
        :arg env: environment model, i.e. ``'chain'``, ``'reduced'`` or
            ``'sliced'``
        :type env: str
        :return: ANM model
        :rtype: :class:`ANM`
        """
        assert env in ['chain', 'reduced', 'sliced', 'full']
        anm_e = self._anm[env]
        n = self.n_modes
        if anm_e[chID] is None:
            pdb = self.getPDB()
            if env == 'chain':
                ca = pdb.ca[chID]
                self._checkNumCalphas(ca)
                anm = ANM()
                anm.buildHessian(ca)
                anm.calcModes(n_modes=n)
                anm_e[chID] = anm
            elif env == 'reduced':
                system = pdb.ca[chID]
                self._checkNumCalphas(system)
                selstr = f'(not chain `{chID}` and name CA) or '
                selstr += '(resname NE1 name Q1) or '
                selstr += '(nucleic name P C4\' C2)'
                environment = pdb.select(selstr)
                if environment is None:
                    anm = ANM()
                    anm.buildHessian(system, cutoff=15.0, gamma=1.0)
                    anm.calcModes(n_modes=n)
                    anm_e[chID] = anm
                else:
                    anm = envANM()
                    anm.buildHessian(system, environment, cutoff=15.0, gamma=1.0)
                    anm.calcModes(n_modes=n)
                    anm_e[chID] = anm
            else: # env == 'sliced'
                ca = pdb.ca
                self._checkNumCalphas(ca)
                anm_full = ANM()
                anm_full.buildHessian(ca)
                anm_full.calcModes(n_modes=n)
                for c in self.chids:
                    # sel = 'chain ' + c
                    sel = f'chain `{c}`'
                    anm, _ = sliceModel(anm_full, ca, sel)
                    anm_e[c] = anm
        return self._anm[env][chID]
    
    def calcGNMfeatures(self, chain='all', env='chain'):
        LOGGER.timeit("_calcGNMfeatures")
        assert env in ['chain', 'reduced', 'sliced', 'full']
        # list of features to be computed
        features = ['GNM_Ventropy', 'GNM_rmsf_overall',
                    'GNM_Eigval1', 'GNM_Eigval2', 'GNM_Eigval5_1',
                    'GNM_SEall', 'GNM_SE20',
                    'GNM_V1', 'GNM_rankV1', 
                    'GNM_V2', 'GNM_rankV2',
                    'GNM_co_rank', 'GNM_displacement',
                    'GNM_MC1', 'GNM_MC2']
        features = [f+'_'+env for f in features]
        # compute features (if not precomputed)
        if chain == 'all':
            chain_list = self.chids
        else:
            chain_list = [chain, ]
        pdb = self.getPDB()
        # Loop over chains
        for chID in chain_list:
            d = self.feats[chID]
            # Check if all features are already computed
            if all([f in d for f in features]):
                if all([not isinstance(d[f], str) for f in features]):
                    continue
            # Compute GNM
            try:
                gnm = self.calcGNM(chID, env=env)
            except Exception as e:
                msg = traceback.format_exc()
                LOGGER.warn(msg)
                for f in features:
                    d[f] = str(e)
                continue

            # Compute the rest
            try:
                n = gnm.numAtoms()
                eigvals = gnm.getEigvals()
                eigvecs = gnm.getEigvecs()
                eigval5_1 = np.log10(eigvals[4]) - np.log10(eigvals[0])
                rmsf_overall = np.sqrt(np.sum(1./eigvals) / n)
                cov_matrix = self._cov_matrix(eigvecs, eigvals)
                SEall = calcSpectralEntropy(gnm, n_modes='all')
                SE20 = calcSpectralEntropy(gnm, n_modes=int(round(0.2*gnm.numModes(), 0)))
                Ventropy = (0.5*n) + (((n * np.log(2*np.pi)) - np.sum(np.log(eigvals))) * 0.5) 
                V1 = np.abs(eigvecs[:, 0])                          
                rankV1 = self._rankGVecs(eigvecs[:, 0])             
                V2 = np.abs(eigvecs[:, 1])                          
                rankV2 = self._rankGVecs(eigvecs[:, 1])             
                displacement = cov_matrix[0]                        
                co_rank = cov_matrix[1]                             
                MC1 = (1/eigvals[0]) * np.abs(eigvecs[:, 0])        
                MC2 = (1/eigvals[1]) * np.abs(eigvecs[:, 1])        
                
                # Fill features
                if env != 'full':
                    d['GNM_rmsf_overall_'+env]  = rmsf_overall             # protein feature
                    d['GNM_Ventropy_'+env]      = Ventropy                 # protein feature
                    d['GNM_Eigval1_'+env]       = eigvals[0]               # protein feature
                    d['GNM_Eigval2_'+env]       = eigvals[1]               # protein feature
                    d['GNM_Eigval5_1_'+env]     = eigval5_1                # protein feature
                    d['GNM_SEall_'+env]         = SEall                    # protein feature
                    d['GNM_SE20_'+env]          = SE20                     # protein feature
                    d['GNM_V1_'+env]            = V1
                    d['GNM_rankV1_'+env]        = rankV1
                    d['GNM_V2_'+env]            = V2
                    d['GNM_rankV2_'+env]        = rankV2
                    d['GNM_displacement_'+env]  = displacement
                    d['GNM_co_rank_'+env]       = co_rank
                    d['GNM_MC1_'+env]           = MC1
                    d['GNM_MC2_'+env]           = MC2
                else: # env == 'full'
                    for c in self.chids:
                        sel = f'chain `{c}`'
                        chid_which, sel = sliceAtoms(pdb.ca, sel)
                        chain_length = len(chid_which)
                        for f in features:
                            self.feats[c][f] = np.full(chain_length, np.nan)
                        if self.format == 'af':
                            which, sel = sliceAtoms(pdb.ca.select('beta >= 50'), sel)
                        else:
                            which = chid_which
                        # protein features
                        self.feats[c]['GNM_rmsf_overall_'+env]  = rmsf_overall
                        self.feats[c]['GNM_Ventropy_'+env]      = Ventropy
                        self.feats[c]['GNM_Eigval1_'+env]       = eigvals[0]
                        self.feats[c]['GNM_Eigval2_'+env]       = eigvals[1]
                        self.feats[c]['GNM_Eigval5_1_'+env]     = eigval5_1
                        self.feats[c]['GNM_SEall_'+env]         = SEall
                        self.feats[c]['GNM_SE20_'+env]          = SE20
                        # residue features
                        self.feats[c]['GNM_V1_'+env]            = V1[which]
                        self.feats[c]['GNM_rankV1_'+env]        = rankV1[which]
                        self.feats[c]['GNM_V2_'+env]            = V2[which]
                        self.feats[c]['GNM_rankV2_'+env]        = rankV2[which]
                        self.feats[c]['GNM_displacement_'+env]  = displacement[which]
                        self.feats[c]['GNM_co_rank_'+env]       = co_rank[which]
                        self.feats[c]['GNM_MC1_'+env]           = MC1[which]
                        self.feats[c]['GNM_MC2_'+env]           = MC2[which]
            except Exception as e:
                msg = traceback.format_exc()
                LOGGER.warn(msg)
                for f in features:
                    d[f] = str(e)
        
        LOGGER.report('GNM features computed in %.2fs.', "_calcGNMfeatures")

    def calcANMfeatures(self, chain='all', env='chain',
                    ANM_PRS=True, stiffness=True):
        """Computes ANM-based features.

        :arg chain: chain identifier
        :type chain: str
        :arg env: environment model, i.e. ``'chain'``, ``'reduced'`` or
            ``'sliced'``
        :type env: str
        :arg ANM_PRS: whether or not to compute features based on Perturbation
            Response Scanning analysis
        :type ANM_PRS: bool
        :arg stiffness: whether or not to compute stiffness with MechStiff
        :type stiffness: bool
        """
        
        LOGGER.timeit("_calcANMfeatures")
        features = []
        assert env in ['chain', 'reduced', 'sliced', 'full']
        for k in ANM_PRS, stiffness:
            assert type(k) is bool
        # list of features to be computed
        if ANM_PRS:
            features += ['ANM_effectiveness', 'ANM_sensitivity']
        if stiffness:
            features += ['ANM_stiffness']
        features = [f+'_'+env for f in features]
        # compute features (if not precomputed)
        if chain == 'all':
            chain_list = self.chids
        else:
            chain_list = [chain, ]
        # Loop over chains
        for chID in chain_list:
            d = self.feats[chID]
            if all([f in d for f in features]):
                if all([isinstance(d[f], np.ndarray) for f in features]):
                    continue
            # Compute ANM
            try:
                anm = self.calcANM(chID, env=env)
            except Exception as e:
                msg = traceback.format_exc()
                if (isinstance(e, MemoryError)):
                    msg = 'MemoryError' + msg
                else:
                    msg = str(e) + msg
                LOGGER.warn(msg)
                for f in features:
                    d[f] = str(e)
                continue
            # Compute PRS
            key_eff = 'ANM_effectiveness_' + env
            if key_eff in features and key_eff not in d:
                key_sns = 'ANM_sensitivity_' + env
                try:
                    prs_mtrx, eff, sns = calcPerturbResponse(anm)
                    d[key_eff] = eff
                    d[key_sns] = sns
                except Exception as e:
                    msg = traceback.format_exc()
                    LOGGER.warn(msg)
                    d[key_eff] = str(e)
                    d[key_sns] = str(e)
            # Compute stiffness
            key_stf = 'ANM_stiffness_' + env
            if key_stf in features and key_stf not in d:
                try:
                    pdb = self.getPDB()
                    ca = pdb[chID].ca
                    stiff_mtrx = calcMechStiff(anm, ca)
                    d[key_stf] = np.mean(stiff_mtrx, axis=0)
                except Exception as e:
                    msg = traceback.format_exc()
                    LOGGER.warn(msg)
                    d[key_stf] = str(e)
        
        LOGGER.report('ANM features computed in %.2fs.', "_calcANMfeatures")
        return
    
    #####################################
    # STR features

    def calcRONNfeature(self, chain="all"):
        if chain == "all":
            chain_list = self.chids
        else:
            chain_list = [chain, ]
        pdb = self.getPDB()
        for chID in chain_list:
            d = self.feats[chID]
            if 'IDRs' in d:
                if isinstance(d['IDRs'], np.ndarray):
                    continue
            seq = pdb[chID].getSequence()
            try:
                IDRs = ronn.calc_ronn(seq)
                d['IDRs'] = IDRs
            except Exception as e:
                msg = traceback.format_exc()
                LOGGER.warn(msg)
                d['IDRs'] = str(e)

    def calcSFfeatures(self):
        pdb = self.getPDB()
        for chID in self.chids:
            d = self.feats[chID]
            if all([f in d for f in ['SF1', 'SF2', 'SF3']]):
                if all([not isinstance(d[f], str) for f in ['SF1', 'SF2', 'SF3']]):
                    continue
        try:
            sf1, sf2, sf3 = calcShapeFactors(pdb.ca)
            for chID in self.chids:
                self.feats[chID]['SF1'] = sf1
                self.feats[chID]['SF2'] = sf2
                self.feats[chID]['SF3'] = sf3

        except Exception as e:
            msg = traceback.format_exc()
            LOGGER.warn(msg)
            for chID in self.chids:
                self.feats[chID]['SF1'] = str(e)
                self.feats[chID]['SF2'] = str(e)
                self.feats[chID]['SF3'] = str(e)
    
    def calcCLfeature(self, chain="all"):
        """Calculate chain length feature."""
        pdb = self.getPDB()
        protein_length = pdb.ca.numAtoms()
        if chain == 'all':
            chain_list = self.chids
        else:
            chain_list = [chain, ]
        for chID in chain_list:
            d = self.feats[chID]
            chain_length = pdb[chID].ca.numAtoms()
            d['chain_length'] = chain_length
            d['protein_length'] = protein_length
    
    def calcAGfeatures(self, chain="all", refresh=False):
        features = ['AG1', 'AG3', 'AG5', 'ACR']
        pdb = self.getPDB()
        if chain == 'all':
            chain_list = self.chids
        else:
            chain_list = [chain, ]
        # Loop over chains
        for chID in chain_list:
            d = self.feats[chID]
            if not refresh:
                if all([f in d for f in features]):
                    if all([not isinstance(d[f], str) for f in features]):
                        continue
            try:
                ag = calcAG(pdb, chain=chID)
                for f in features:
                    if f != 'ACR':
                        d[f] = ag[f]
                    else:
                        d[f] = np.mean(ag['AG1'])
                        # d[f] = np.ones(chain_length) * np.mean(ag['AG1'])
            except Exception as e:
                msg = traceback.format_exc()
                LOGGER.warn(msg)
                for f in features:
                    d[f] = str(e)
    
    def calcRGandDcomfeatures(self):
        features = ['Rg', 'Dcom']
        pdb = self.getPDB()

        if all([f in self.feats[chID] for chID in self.chids for f in features]):
            if all([not isinstance(self.feats[chID][f], str) for chID in self.chids for f in features]):
                return
        try:
            rg, dcom = calcRGandDcom(pdb)
            for c in self.chids:
                sel = f'chain `{c}`'
                which, sel = sliceAtoms(pdb.ca, sel)
                # chain_length = len(which)
                # self.feats[c]['Rg'] = np.ones(chain_length) * rg
                self.feats[c]['Rg'] = rg
                self.feats[c]['Dcom'] = dcom[which]
        except Exception as e:
            msg = traceback.format_exc()
            LOGGER.warn(msg)
            for chID in self.chids:
                d = self.feats[chID]
                for f in features:
                    d[f] = str(e)
                    
    def calcDSSfeatures(self):
        pdb = self.getPDB()
        ca = pdb.ca.copy()
        for chID in self.chids:
            d = self.feats[chID]
            if 'SSbond' in d:
                if isinstance(d['SSbond'], np.ndarray):
                    return
        try:
            dss = calcDisulfideBonds(pdb.protein, distA=2.4)
            dss_arr = np.zeros(ca.numAtoms())
            for bond in dss:
                # chid, resnum, icode
                atom1 = (bond[5], int(bond[1]), bond[2])
                atom2 = (bond[11], int(bond[7]), bond[8])
                atom1_idx = ca[atom1].getIndices()[0]
                atom2_idx = ca[atom2].getIndices()[0]
                dss_arr[atom1_idx] = 1
                dss_arr[atom2_idx] = 1
            ca.setData('DisulfideBonds', dss_arr)
            for chID in self.chids:
                self.feats[chID]['SSbond'] = ca[chID].getData('DisulfideBonds')
        except Exception as e:
            msg = traceback.format_exc()
            LOGGER.warn(msg)
            for chID in self.chids:
                self.feats[chID]['SSbond'] = str(e)

    def calcDSSPfeatures(self, chain='all'):
        features = ['loop_percent', 'helix_percent', 'sheet_percent', 'dssp']
        if chain == 'all':
            chain_list = self.chids
        else:
            chain_list = [chain, ]
        # Loop over chains
        for chID in chain_list:
            d = self.feats[chID]
            if all([f in d for f in features]):
                if all([not isinstance(d[f], str) for f in features]):
                    continue
            try:
                l, s, h, dssp = calcSecondary(self.pdbPath, chain=chID)
                d['loop_percent'] = l
                d['sheet_percent'] = s
                d['helix_percent'] = h
                d['dssp'] = dssp
            except Exception as e:
                msg = traceback.format_exc()
                LOGGER.warn(msg)
                for f in features:
                    d[f] = str(e)

    def calcSASAfeatures(self, chain='all', complex=False):
        features = ['SASA', 'SASA_in_complex', 'deltaSASA']
        if chain == 'all':
            chain_list = self.chids
        else:
            chain_list = [chain, ]
        # Loop over chains
        for chID in chain_list:
            d = self.feats[chID]
            if 'SASA' in d:
                if isinstance(d['SASA'], np.ndarray):
                    continue
            try:
                ag = calcAccessibility(self.pdbPath, chain=chID)
                d['SASA'] = ag.ca[chID].getData('naccess_aa_rel')
                if len(self.chids) == 1:
                    d['SASA_in_complex'] = d['SASA']
                    d['deltaSASA'] = np.zeros(d['SASA'].shape)
            except Exception as e:
                msg = traceback.format_exc()
                LOGGER.warn(msg)
                d['SASA'] = str(e)

        # Calculate SASA in complex and deltaSASA
        if complex:
            try:
                if not all(['SASA_in_complex' in self.feats[chID] for chID in self.chids]):
                    ag = calcAccessibility(self.pdbPath, chain="all")
                    for chID in self.chids:
                        d = self.feats[chID]
                        d['SASA_in_complex'] = ag.ca[chID].getData('naccess_aa_rel')
                else:
                    if not all([isinstance(self.feats[chID]['SASA_in_complex'], np.ndarray) for chID in self.chids]):
                        ag = calcAccessibility(self.pdbPath, chain="all")
                        for chID in self.chids:
                            d = self.feats[chID]
                            d['SASA_in_complex'] = ag.ca[chID].getData('naccess_aa_rel')
                for chID in chain_list:
                    d = self.feats[chID]
                    d['deltaSASA'] = d['SASA_in_complex'] - d['SASA']
            except Exception as e:
                msg = traceback.format_exc()
                LOGGER.warn(msg)
                for chID in self.chids:
                    d = self.feats[chID]
                    d['SASA_in_complex'] = str(e)
                    d['deltaSASA'] = str(e)

    def calcHbondfeature(self, chain_list='all'):
        if chain_list == 'all':
            chain_list = self.chids
        chid_to_run = []
        for chID in chain_list:
            d = self.feats[chID]
            if 'Hbond' in d:
                if isinstance(d['Hbond'], np.ndarray):
                    continue
                else:
                    chid_to_run.append(chID)
            else:
                chid_to_run.append(chID)
        if chid_to_run == []:
            return
        try:
            ag = calcHbond(self.pdbPath, chain_list=chid_to_run)
            for chID in chid_to_run:
                d = self.feats[chID]
                d['Hbond'] = ag.ca[chID].getData('hbond')
        except Exception as e:
            msg = traceback.format_exc()
            LOGGER.warn(msg)
            for chID in chid_to_run:
                d = self.feats[chID]
                d['Hbond'] = str(e)

    def calcPropKafeature(self, chain='all'):
        if chain == 'all':
            chain_list = self.chids
        else:
            chain_list = [chain, ]
        # Loop over chains
        for chID in chain_list:
            d = self.feats[chID]
            if 'charge_pH7' in d:
                if isinstance(d['charge_pH7'], np.ndarray):
                    continue
            try:
                ag = calcChargepH7(self.pdbPath, chain=chID)
                d['charge_pH7'] = ag.ca[chID].getData('charge_pH7')
            except Exception as e:
                msg = traceback.format_exc()
                LOGGER.warn(msg)
                d['charge_pH7'] = str(e)

    def calcConSurffeatures(self, chain='all'):
        """
        For ConSurf features, we do not use pdbpath as input, but the parsed asymmetric unit structure.
        Reason is that ConSurf provides calculations for the asymmetric unit, not the biological unit.
        Especially when you do searching the chainID in https://consurfdb.tau.ac.il/
        """
        features = ['consurf', 'ACNR', 'consurf_color']
        if chain == 'all':
            chain_list = self.chids
        else:
            chain_list = [chain, ]
        for chID in chain_list:
            d = self.feats[chID]
            if all([f in d for f in features]):
                if all([not isinstance(d[f], str) for f in features]):
                    continue
            try: 
                if self.format == 'custom' or self.format == 'af':
                    # If the format is custom or af, we need to use the pdbPath
                    # And calculate ConSurf based on stand_alone_consurf
                    f = calcConSurf(self.pdbPath, chID, folder=self.job_directory)  
                else:
                    f = calcConSurf(self.pdbID, chID, folder=self.job_directory)
                d['consurf'] = f['consurf']
                d['ACNR'] = f['ACNR']
                d['consurf_color'] = f['consurf_color']
            except Exception as e:
                msg = traceback.format_exc()
                LOGGER.warn(msg)
                for f in features:
                    d[f] = str(e)

    def calcDELTA_Rg_SASA_ACR_Hbond_DSS_features(self, chids, resids, wt_aas, mut_aas, 
        sel_feats=['DELTA_Rg', 'DELTA_Dcom', 'DELTA_SASA', 'DELTA_ACR', 
                       'DELTA_Hbond', 'DELTA_DSS', 'DELTA_charge_pH7']):
        # Merge if either DELTA_Rg or DELTA_Dcom is selected
        if {'DELTA_Rg', 'DELTA_Dcom'}.intersection(set(sel_feats)):
            sel_feats = list(set(sel_feats) | {'DELTA_Rg', 'DELTA_Dcom'})
        _dtype = np.dtype([(f, 'f') for f in sel_feats])
        f = np.full(len(chids), np.nan, dtype=_dtype)
        pdb = self.getPDB()

        # In pdb contains only CA atoms, DELTA features are set to 0
        if pdb.protein.numAtoms() == pdb.ca.numAtoms():
            LOGGER.warn("PDB contains only CA atoms. DELTA features are set to 0.")
            for name in sel_feats:
                f[name] = 0
            return f

        for i, (chid, resid, wt_aa, mut_aa) in enumerate(zip(chids, resids, wt_aas, mut_aas)):
            chain = pdb.ca[chid].copy()
            d = self.feats[chid]
            try:
                # Make the file for the mutation
                wtpath = writePDB(os.path.join(self.folder, f'{self.pdbID}_{chid}.pdb'), pdb.protein[chid].copy())
                mutpath = createMutationfile(wtpath, chid, mutation=f'{wt_aa}{resid}{mut_aa}')
            except Exception as e:
                msg = traceback.format_exc()
                LOGGER.warn(f"Error in creating mutation file for {chid} {resid} {wt_aa} {mut_aa}")
                LOGGER.warn(msg)
                continue
            # Find index
            indices = self._findIndex(chid, resid)
            # Calculate the features
            if {'DELTA_Rg', 'DELTA_Dcom'}.intersection(set(sel_feats)):
                if not all([f in d for f in ['Rg', 'Dcom']]):
                    self.calcRGandDcomfeatures()
                else:
                    if not all([isinstance(d[f], np.ndarray) for f in ['Rg', 'Dcom']]):
                        self.calcRGandDcomfeatures()
                try:
                    rg, dcom = calcRGandDcom(mutpath)
                    f[i]['DELTA_Rg'] = rg - self.feats[chid]['Rg']
                    f[i]['DELTA_Dcom'] = dcom[indices[0]] - self.feats[chid]['Dcom'][indices[0]]
                except Exception as e:
                    msg = traceback.format_exc()
                    LOGGER.warn(msg)
            if 'DELTA_SASA' in sel_feats:
                if not 'SASA' in d:
                    self.calcSASAfeatures(chain=chid)
                else:
                    if not isinstance(d['SASA'], np.ndarray):
                        self.calcSASAfeatures(chain=chid)
                try:
                    sasa = calcAccessibility(mutpath, chain=chid)
                    f[i]['DELTA_SASA'] = sasa.ca[chid].getData('naccess_aa_rel')[indices[0]] - \
                        self.feats[chid]['SASA'][indices[0]]
                except Exception as e:
                    msg = traceback.format_exc()
                    LOGGER.warn(msg)
            if 'DELTA_ACR' in sel_feats:
                if not 'ACR' in d:
                    self.calcAGfeatures(chain=chid)
                else:
                    if not isinstance(d['ACR'], np.ndarray):
                        self.calcAGfeatures(chain=chid)
                try:
                    ag = calcAG(mutpath, chain=chid)
                    f[i]['DELTA_ACR'] = np.mean(ag['AG1']) - self.feats[chid]['ACR']
                except Exception as e:
                    msg = traceback.format_exc()
                    LOGGER.warn(msg)
            if 'DELTA_Hbond' in sel_feats:
                if not 'Hbond' in d:
                    self.calcHbondfeature(chain_list=[chid])
                else:
                    if not isinstance(d['Hbond'], np.ndarray):
                        self.calcHbondfeature(chain_list=[chid])
                try:
                    ag = calcHbond(mutpath, chain_list=[chid])
                    f[i]['DELTA_Hbond'] = ag.ca[chid].getData('hbond')[indices[0]] - \
                        self.feats[chid]['Hbond'][indices[0]]
                except Exception as e:
                    msg = traceback.format_exc()
                    LOGGER.warn(msg)
            if 'DELTA_DSS' in sel_feats:
                if not 'SSbond' in d:
                    self.calcDSSfeatures()
                else:
                    if not isinstance(d['SSbond'], np.ndarray):
                        self.calcDSSfeatures()
                if mut_aa != 'C':
                    f[i]['DELTA_DSS'] = 0 - self.feats[chid]['SSbond'][indices[0]]
                else:
                    try:       
                        dss = calcDisulfideBonds(parsePDB(mutpath), distA=2.4)
                        dss_arr = np.zeros(len(self.feats[chid]['SSbond']))
                        for bond in dss:
                            # chid, resnum, icode
                            atom1 = (bond[5], int(bond[1]), bond[2])
                            atom2 = (bond[11], int(bond[7]), bond[8])
                            atom1_idx = chain[atom1].getIndices()[0]
                            atom2_idx = chain[atom2].getIndices()[0]
                            dss_arr[atom1_idx] = 1
                            dss_arr[atom2_idx] = 1
                        f[i]['DELTA_DSS'] = dss_arr[indices[0]] - self.feats[chid]['SSbond'][indices[0]]
                    except Exception as e:
                        msg = traceback.format_exc()
                        LOGGER.warn(msg)
            if 'DELTA_charge_pH7' in sel_feats:
                if not 'charge_pH7' in d:
                    self.calcPropKafeature(chain=chid)
                else:
                    if not isinstance(d['charge_pH7'], np.ndarray):
                        self.calcPropKafeature(chain=chid)
                try:
                    ag = calcChargepH7(mutpath, chain=chid)
                    f[i]['DELTA_charge_pH7'] = ag.ca[chid].getData('charge_pH7')[indices[0]] - \
                        self.feats[chid]['charge_pH7'][indices[0]]
                except Exception as e:
                    msg = traceback.format_exc()
                    LOGGER.warn(msg)
            
            # if exist wtpath and mutpath, remove them
            try: 
                os.remove(wtpath)
                os.remove(mutpath)
            except:
                pass
        return f

    def _cov_matrix(self, GVecs, GVals):
        N = GVecs.shape[0]
        m = GVecs @ np.diag(1./GVals) @ GVecs.T
        displacement = np.diag(m)
        displacement = np.abs(displacement) 
        rank = np.unique(displacement, return_inverse=True)[1]
        rank = np.max(rank) - rank + 1
        rank = 1 - (rank / N)
        return displacement, rank

    def _rankGVecs(self, GVec):
        N = GVec.shape[0]
        GVec = np.abs(GVec)
        rank_GVec = GVec.argsort().argsort()
        rank_GVec = np.max(rank_GVec) - rank_GVec + 1
        rank_GVec = 1 - (rank_GVec / N)
        return rank_GVec

    def _findIndex(self, chain, resid):
        indices = np.where(self.resids[chain] == resid)[0] # Residue numbers (resids/resnums)
        if len(indices) > 1:
            LOGGER.warn('Multiple ({}) residues with resid {} found.'
                        .format(len(indices), resid))
        return indices
    
    def calcFeatures(self, chids: list, resids: list, 
                     wt_aas: list, mut_aas: list, sel_feats: list):
        """Selects features for a given chain and residue."""
        # sel_feats must be within the list of available features: ANM_FEATURES + GNM_FEATURES
        if not set(sel_feats).issubset(set(PDB_FEATS)):
            invalid_feats = set(sel_feats) - set(PDB_FEATS)
            raise ValueError(f'Invalid features: {invalid_feats}')
        chain_list = np.unique(chids)
        chain_list = [c for c in chain_list if c in self.chids]

        # Initialize features
        f_dtype = np.dtype([(f, 'f') for f in sel_feats])
        f = np.full(len(resids), np.nan, dtype=f_dtype)

        LOGGER.timeit('_calcFeatures')
        # Calculate GNM and ANM features
        for env in ['chain', 'reduced', 'sliced', 'full']:
            s = '_' + env
            l = [f.replace(s, '') for f in sel_feats if f.endswith(s)]
            if l == []:
                continue
            ANM_PRS, stiffness = (False,)*2
            if 'ANM_effectiveness' in l or 'ANM_sensitivity' in l:
                ANM_PRS = True
            if 'ANM_stiffness' in l:
                stiffness = True
            for chain in chain_list:
                if chain in self.chids:
                    if any('GNM' in s for s in l):
                        self.calcGNMfeatures(chain, env=env)
                    if any('ANM' in s for s in l):
                        self.calcANMfeatures(chain, env=env, ANM_PRS=ANM_PRS,
                                            stiffness=stiffness)
                else:
                    LOGGER.warn(f'Chain {chain} not found.')
        # Calculate chain length
        if {'chain_length', 'protein_length'}.intersection(set(sel_feats)):
            for chain in chain_list:
                self.calcCLfeature(chain)
        # Calculate RONN feature
        if 'IDRs' in sel_feats:
            for chain in chain_list:
                self.calcRONNfeature(chain)
        # Calculate Disulfide Bonds
        if 'SSbond' in sel_feats:
            self.calcDSSfeatures()
        # Calculate Shape Factors
        if {'SF1', 'SF2', 'SF3'}.intersection(set(sel_feats)):
            self.calcSFfeatures()
        # Calculate AG features
        if {'AG1', 'AG3', 'AG5', 'ACR'}.intersection(set(sel_feats)):
            for chain in chain_list:
                self.calcAGfeatures(chain)
        # Calculate RG and Dcom features
        if {'Rg', 'Dcom'}.intersection(set(sel_feats)):
            self.calcRGandDcomfeatures()
        # Calculate DSSP features
        if {'loop_percent', 'helix_percent', 'sheet_percent', 'dssp'}.intersection(set(sel_feats)):
            for chain in chain_list:
                self.calcDSSPfeatures(chain)
        # Calculate SASA features
        if 'SASA' in sel_feats and {'SASA_in_complex', 'deltaSASA'}.intersection(set(sel_feats)):
            for chain in chain_list:
                self.calcSASAfeatures(chain, complex=True)
        elif 'SASA' in sel_feats and not {'SASA_in_complex', 'deltaSASA'}.intersection(set(sel_feats)):
            for chain in chain_list:
                self.calcSASAfeatures(chain, complex=False)
        # Calculate Hbond feature
        if 'Hbond' in sel_feats:
            self.calcHbondfeature(chain_list)
        # Calculate PropKa feature
        if 'charge_pH7' in sel_feats:
            for chain in chain_list:
                self.calcPropKafeature(chain)
        # Calculate BJCE features
        if {'wtBJCE', 'mutBJCE', 'deltaBJCE'}.intersection(set(sel_feats)):
            pdb = self.getPDB()
            bjce = calcBJCEnergy(pdb, chids, resids, wt_aas, mut_aas)
            for name in ['wtBJCE', 'mutBJCE', 'deltaBJCE']:
                if name in sel_feats:
                    f[name] = bjce[name]
        # Calculate Lside features
        if {'Lside', 'deltaLside'}.intersection(set(sel_feats)):
            Lside, deltaLside = calcLside(wt_aas, mut_aas)
            f['Lside'] = Lside
            f['deltaLside'] = deltaLside
        # Calculate ConSurf features
        if {'consurf', 'ACNR', 'consurf_color'}.intersection(set(sel_feats)):
            for chain in chain_list:
                self.calcConSurffeatures(chain)
        # Calculate DELTA features
        if {'DELTA_Rg', 'DELTA_Dcom', 'DELTA_SASA', 'DELTA_charge_pH7',
            'DELTA_ACR', 'DELTA_Hbond', 'DELTA_DSS'}.intersection(set(sel_feats)):
            DELTA_sel_feats = [fn for fn in sel_feats if fn.startswith('DELTA_')]
            delta = self.calcDELTA_Rg_SASA_ACR_Hbond_DSS_features(chids, resids, wt_aas, mut_aas, DELTA_sel_feats)
            for name in DELTA_sel_feats:
                f[name] = delta[name]
        # Select features
        for i, (chid, resid) in enumerate(zip(chids, resids)):
            if chid not in self.chids:
                LOGGER.warn('Chain {} not found.'.format(chid))
                continue
            if resid not in self.resids[chid]:
                LOGGER.warn('Residue {} not found in chain {}.'.format(resid, chid))
                continue
            d = self.feats[chid]
            indices = self._findIndex(chid, resid)
            for name in sel_feats:
                if name in [
                    'wtBJCE', 'mutBJCE', 'deltaBJCE', 'Lside', 'deltaLside', 
                    'DELTA_Rg', 'DELTA_Dcom', 'DELTA_SASA', 'DELTA_ACR', 'DELTA_Hbond', 
                    'DELTA_DSS', 'DELTA_charge_pH7']:
                    continue
                if not isinstance(d[name], str):
                    # protein features
                    if name in [
                        'SF1', 'SF2', 'SF3', 'chain_length', 'protein_length', 'ACR',
                        'Rg', 'loop_percent', 'sheet_percent', 'helix_percent',
                        'GNM_rmsf_overall_chain', 'GNM_Ventropy_chain', 'GNM_Eigval1_chain', 'GNM_Eigval2_chain', 
                        'GNM_Eigval5_1_chain', 'GNM_SEall_chain', 'GNM_SE20_chain',
                        'GNM_rmsf_overall_reduced', 'GNM_Ventropy_reduced', 'GNM_Eigval1_reduced', 'GNM_Eigval2_reduced', 
                        'GNM_Eigval5_1_reduced', 'GNM_SEall_reduced', 'GNM_SE20_reduced',
                        'GNM_rmsf_overall_sliced', 'GNM_Ventropy_sliced', 'GNM_Eigval1_sliced', 'GNM_Eigval2_sliced', 
                        'GNM_Eigval5_1_sliced', 'GNM_SEall_', 'GNM_SE20_sliced',
                        'GNM_rmsf_overall_full', 'GNM_Ventropy_full', 'GNM_Eigval1_full', 'GNM_Eigval2_full', 
                        'GNM_Eigval5_1_full', 'GNM_SEall_full', 'GNM_SE20_full',
                        ]:
                        f[name][i] = d[name]
                    else:
                        # residue features
                        f[name][i] = d[name][indices[0]]
        LOGGER.report('PDB features for {} computed in %.2fs.'.format(self.pdbID), '_calcFeatures')
        return f

def calcPDBfeatures(mapped_SAVs, custom_PDB=None, refresh=False, 
                    sel_feats=PDB_FEATS, withSAV=False, **kwargs):
    """Computes structural and dynamics features from PDB structures.
    mapped_SAVs: numpy array or pandas DataFrame
        columns: 
            'Unique_SAV_coords', 'Asymmetric_PDB_coords', 'Asymmetric_PDB_length',
            'OPM_PDB_coords', 'BioUnit_PDB_coords'
    custom_PDB: str
        custom PDB file path, unzipped
    refresh: bool
        whether to refresh precomputed features
    sel_feats: list
        selected features
    folder: str
        folder to save PDB features


    format is defined as follows:
        ### if custom_PDB is available:
            - custom_PDB is a structure file
                - af: first line contains "alphafoldserver.com/output-terms" (cif file)
                - custom: not alphafold
            - custom_PDB is a PDBID 
                - opm: OPM PDB coordinates are available
                - bas: BioUnit PDB coordinates are available
                - asu: Asymmetric PDB coordinates are available
        ### if custom_PDB is not available:
            - af: PDBID starts with AF-
            - opm: OPM PDB coordinates are available
            - bas: BioUnit PDB coordinates are available
            - asu: Asymmetric PDB coordinates are available
    """
    LOGGER.info('Computing strutural and dynamics features from PDB structures...')
    LOGGER.timeit('_calcPDBfeatures')
    # Convert to numpy array
    if isinstance(mapped_SAVs, pd.DataFrame):
        mapped_SAVs = mapped_SAVs.to_records(index=False)
    elif isinstance(mapped_SAVs, np.ndarray) and mapped_SAVs.shape == ():
        mapped_SAVs = np.array([mapped_SAVs, ])
    nSAVs = mapped_SAVs.shape[0]
    # Initialize features
    _dtype = np.dtype([(f, 'f') for f in sel_feats])
    features = np.full(nSAVs, np.nan, dtype=_dtype)
    
    # Group SAVs by PDB ID and format
    groups = defaultdict(lambda: defaultdict(list))
    # Given custom_PDB file
    if custom_PDB is not None and os.path.isfile(custom_PDB):
        custom_PDB = os.path.abspath(custom_PDB)
        pdbID = custom_PDB.split('/')[-1]
        # Check if custom_PDB is a alphafold structure
        if verifyAF(custom_PDB):
            LOGGER.info(f"Custom PDB {custom_PDB} is an alphafold structure.")
            for i, SAV in enumerate(mapped_SAVs):
                if "Cannot map" not in SAV['Asymmetric_PDB_coords']:
                    groups[pdbID]['af'].append(i)
        else: # Not alphafold --> Assign custom
            for i, SAV in enumerate(mapped_SAVs):
                if "Cannot map" not in SAV['Asymmetric_PDB_coords']:
                    groups[pdbID]['custom'].append(i)
    else: # This includes custom_PDB as pdbID
        for i, SAV in enumerate(mapped_SAVs):
            pdb_len = int(SAV['Asymmetric_PDB_length'])
            if pdb_len == 0:
                continue
            pdbID = SAV['Asymmetric_PDB_coords'].split()[0]
            if pdbID.startswith("AF-"):
                groups[pdbID]['af'].append(i)
            elif "Cannot map" not in SAV['OPM_PDB_coords']:
                groups[pdbID]['opm'].append(i)
            elif "Cannot map" not in SAV['BioUnit_PDB_coords']:
                assemblyID = SAV['BioUnit_PDB_coords'].split()[4]
                groups[pdbID][f'bas{assemblyID}'].append(i)
            else:
                groups[pdbID]['asu'].append(i)
    
    # Compute features for each group
    ndone = 0
    for pdbID, formats in groups.items():
        pdbID = pdbID.lower() if len(pdbID) == 4 else pdbID
        LOGGER.info(f"Processing {pdbID}...")
        for format, indices in formats.items():
            ndone += len(indices)
            # Find correct PDB file and format
            try:
                # Non-AF custom PDB
                if format == 'custom':
                    pdbPath = fixPDB(custom_PDB, format, folder=FIX_PDB_DIR, refresh=True)
                    pdb_coords = mapped_SAVs[indices]['Asymmetric_PDB_coords']
                # AF custom PDB file
                elif custom_PDB is not None and format == 'af':
                    pdbPath = custom_PDB
                    pdb_coords = mapped_SAVs[indices]['Asymmetric_PDB_coords']
                # Not custom PDB
                else:
                    pdbPath = fixPDB(pdbID, format, folder=FIX_PDB_DIR, refresh=True)
                    if format == 'asu':
                        pdb_coords = mapped_SAVs[indices]['Asymmetric_PDB_coords']
                    elif format == 'opm':
                        pdb_coords = mapped_SAVs[indices]['OPM_PDB_coords']
                    elif format == 'af':
                        pdb_coords = mapped_SAVs[indices]['Asymmetric_PDB_coords']
                    else:# format.startswith('bas'):
                        pdb_coords = mapped_SAVs[indices]['BioUnit_PDB_coords']
            except Exception as e:
                msg = traceback.format_exc()
                LOGGER.warn(msg)
                continue
            # Check if PDB file exists 
            if not os.path.isfile(pdbPath):
                LOGGER.warning(f"File {pdbPath} not found.")
                continue
            try:
                # Load PDB structure and calculate features
                LOGGER.info(f"Loading PDB {pdbPath}...")
                obj = PDBfeatures(pdbPath, format=format, recover_pickle=not(refresh), **kwargs)
                # if "full" not in obj._gnm:
                    # obj._gnm['full'] = {chID: None for chID in obj.chids}
                # if "full" not in obj._anm:
                    # obj._anm['full'] = {chID: None for chID in obj.chids}
            except Exception as e:
                msg = traceback.format_exc()
                LOGGER.warn(msg)
                obj = str(e)    
            # Extract features for SAVs
            if isinstance(obj, str):
                features[indices] = obj
            else:
                sav_coords = mapped_SAVs[indices]['SAV_coords']
                chids, resids, wt_aas = zip(*[ele.split()[1:4] for ele in pdb_coords])
                resids = list(map(int, resids))
                mut_aas = [ele.split()[3] for ele in sav_coords]
                features[indices] = obj.calcFeatures(chids, resids, wt_aas, mut_aas, sel_feats)
            # Save PDB features
            if isinstance(obj, PDBfeatures):
                obj.savePickle(**kwargs)
            done = ndone / nSAVs
            LOGGER.info(f"PDB features: {ndone}/{nSAVs} SAVs processed [{done:.0%}]")
    if withSAV:
        _dtype = np.dtype([("SAV_coords", "U50")] + _dtype.descr)
        _features = np.zeros(nSAVs, dtype=_dtype)
        _features['SAV_coords'] = mapped_SAVs['SAV_coords']
        for f in sel_feats:
            _features[f] = features[f]
        features = _features
    LOGGER.report('PDB features computed in %.2fs.', '_calcPDBfeatures')
    return features
