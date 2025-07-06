import os 
import logging
import requests
import urllib.request
from .utils.logger import LOGGER

__all__ = ['pdb_summary', 'fetchPDB', 'fetchPDB_BiologicalAssembly']

pdbe_prefix = 'https://www.ebi.ac.uk/pdbe'

def get_url(url):
    """
    Makes a request to a URL. Returns a JSON of the results
    :param str url:
    :return dict:
    """
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        LOGGER.warning("No data retrieved - %s" % response.status_code)
        LOGGER.info("[No data retrieved - %s] %s" % (response.status_code, response.text))
    return {}

def get_content(url):
    """
    Makes a request to a URL. Returns the content of the results
    :param str url:
    :return str:
    """
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        return response.text
    else:
        LOGGER.warning("No data retrieved - %s" % response.status_code)
    return ''

def pdb_summary(pdbID: str):
    """This call provides a summary of properties of a PDB entry, 
    such as the title of the entry, list of depositors, date of deposition, 
    date of release, date of latest revision, experimental method, 
    list of related entries in case split entries, etc.

    Ref: https://www.ebi.ac.uk/pdbe/api/doc/pdb.html
    """
    summary_url = f'{pdbe_prefix}/api/pdb/entry/summary/{pdbID}'
    LOGGER.info(f'> Parse the summary data of {pdbID} from {summary_url}...')
    data = get_url(summary_url)
    try:
        assemblies = data[pdbID.lower()][0]['assemblies']
        n_assemblies = len(assemblies)
        LOGGER.info(f'{pdbID} has {n_assemblies} assembly(assemblies).')
    except KeyError:
        LOGGER.info(f'{pdbID} has no assembly data.')
        LOGGER.info(f'{pdbID} has no assembly data.')
        n_assemblies = 0
    return n_assemblies

def uniprot_sequence(uniprotACC, outdir: str = None):
    "Download a sequence file from UniProt database."
    url = f"https://rest.uniprot.org/uniprotkb/{uniprotACC}.fasta"
    
    if outdir is not None:
        outpath = os.path.join(outdir, f'{uniprotACC}.fasta')
    
        if os.path.exists(outpath):
            return outpath
        LOGGER.info(f'> Download {uniprotACC}.fasta...')
        try:
            urllib.request.urlretrieve(url, outpath)
            LOGGER.info(f'{uniprotACC}.fasta is downloaded.')
            return outpath
        except urllib.error.HTTPError:
            LOGGER.warning(f'{url} does not exist.')
            return None
    else:
        fasta =  get_content(url)
        if fasta is None:
            return None
        fasta_lines = fasta.split('\n')
        fasta_lines = [line.strip() for line in fasta_lines if line.strip() != '']
        seq = ''.join(fasta_lines[1:])
        return seq

def fetch_fasta(accs, **kwargs):
    """
    Fetch one or multiple UniProt sequences and save into a single FASTA file.
    
    Parameters:
    - accs: str or list of str, UniProt accession(s)
    - folder: str, output folder (default: '.')
    - outname: str, output filename (default: 'output.fasta')
    - refresh: bool, whether to force re-download even if file exists (default: False)
    
    Returns:
    - Output FASTA file path
    """
    
    if isinstance(accs, str):
        accs = [accs]
    
    folder = kwargs.get('folder', '.')
    filename = kwargs.get('filename', 'output')
    refresh = kwargs.get('refresh', False)
    os.makedirs(folder, exist_ok=True)
    
    output_path = os.path.join(folder, f'{filename}.fasta')
    if os.path.exists(output_path) and not refresh:
        LOGGER.info(f"{output_path} already exists (use refresh=True to overwrite).")
        return output_path
    
    with open(output_path, 'w') as f_out:
        for acc in accs:
            url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
            response = requests.get(url)
            if response.status_code == 200:
                f_out.write(response.text)
                LOGGER.info(f"Fetched: {acc}")
            else:
                LOGGER.info(f"Failed: {acc} (HTTP {response.status_code})")
    LOGGER.info(f"\n✅ Saved to: {output_path}")
    return output_path

def fetchAF2(acc, **kwargs):
    """Fetch a PDB file from AlphaFold2 database."""
    
    folder = kwargs.get('folder', '.')
    refresh = kwargs.get('refresh', False)
    version = kwargs.get('version', 4)
    os.makedirs(folder, exist_ok=True)

    # Define the URL and output path
    name = f'AF-{acc}-F1-model_v{version}.pdb'
    outpath = os.path.join(folder, name)
    url = f"https://alphafold.ebi.ac.uk/files/{name}"
    
    outpath = os.path.abspath(outpath)
    # Check if the file already exists
    if not refresh:
        if os.path.exists(outpath):
            return outpath
    
    # Fetch the file
    try:
        urllib.request.urlretrieve(url, outpath)
        return outpath
    except Exception as e:
        LOGGER.info(f"Failed to fetch {acc} from AlphaFold2 database {e}.")
        return None

def fetchPDB(pdbID, **kwargs):
    """Fetch a PDB file from RCSB PDB database."""
    
    pdbID = pdbID.lower()
    folder = kwargs.get('folder', '.')
    compressed = kwargs.get('compressed', True)
    format = kwargs.get('format', 'pdb')
    refresh = kwargs.get('refresh', False)
    assert format in ['pdb', 'cif', 'opm'], f"format should be 'pdb', 'cif', or 'opm'."
    os.makedirs(folder, exist_ok=True)

    # Define the URL and output path
    if format == 'pdb':
        outpath = os.path.join(folder, f'{pdbID}.pdb.gz')
        url = f"https://files.rcsb.org/download/{pdbID}.pdb.gz"
    elif format == 'cif':
        outpath = os.path.join(folder, f'{pdbID}.cif.gz')
        url = f"https://files.rcsb.org/download/{pdbID}.cif.gz"
    else: # format == 'opm'
        outpath = os.path.join(folder, f'{pdbID}-opm.pdb')
        url = f"https://opm-assets.storage.googleapis.com/pdb/{pdbID}.pdb"
    
    outpath = os.path.abspath(outpath)
    # Check compressed
    if not compressed and format != 'opm':
        outpath = outpath[:-3] 
        url = url[:-3]
    # Check refresh
    if not refresh:
        if os.path.exists(outpath):
            return outpath
        
    # Fetch the file
    try:
        urllib.request.urlretrieve(url, outpath)
        if format == 'opm':
            # Remove 'END' lines from OPM file
            # > This helps PDBFixer recognize the Dummy atoms in some OPM files 
            with open(outpath, 'r') as file:
                lines = file.readlines()
            lines = [line for line in lines 
                if not line.startswith('END') and not line.startswith('CRYST1')]
            with open(outpath, 'w') as file:
                file.writelines(lines)
        if format == 'pdb' and not compressed:
            # Remove 'CRYST1' lines from PDB file
            # This keeps the PDBFixer from raising an error
            with open(outpath, 'r') as file:
                lines = file.readlines()
            lines = [line for line in lines if not line.startswith('CRYST1')]
            with open(outpath, 'w') as file:
                file.writelines(lines)
        elif format == 'pdb' and compressed:
            # Remove 'CRYST1' lines from PDB file
            # This keeps the PDBFixer from raising an error
            with open(outpath, 'rb') as file:
                lines = file.readlines()
            lines = [line for line in lines if not line.startswith(b'CRYST1')]
            with open(outpath, 'wb') as file:
                file.writelines(lines)
        return outpath
    except Exception as e:
        # msg = traceback.format_exc()
        # LOGGER.info(msg)
        LOGGER.info(f"Failed to fetch {pdbID} from RCSB PDB database {e}.")
        if format != 'cif':
            LOGGER.info(f"Fetch cif file instead.")
            return fetchPDB(pdbID, format='cif', folder=folder, compressed=compressed, refresh=refresh)
        return None

def fetchPDB_BiologicalAssembly(pdbID, assemblyID=1, **kwargs):
    """Fetch a PDB file from RCSB PDB database."""
    
    pdbID = pdbID.lower()
    assemblyID = int(assemblyID)
    folder = kwargs.get('folder', '.')
    compressed = kwargs.get('compressed', True)
    format = kwargs.get('format', 'pdb')
    assert format in ['pdb', 'cif'], f"format should be 'pdb' or 'cif'."
    assert assemblyID > 0, f"assemblyID should be greater than 0."
    os.makedirs(folder, exist_ok=True)  

    if format == 'pdb':
        outpath = os.path.join(folder, f'{pdbID}.pdb{assemblyID}.gz')
        url = f"https://files.rcsb.org/download/{pdbID}.pdb{assemblyID}.gz"
    else: # format == 'cif'
        outpath = os.path.join(folder, f'{pdbID}-assembly{assemblyID}.cif.gz')
        url = f"https://files.rcsb.org/download/{pdbID}-assembly{assemblyID}.cif.gz"
    outpath = os.path.abspath(outpath)

    # Check if the file already exists
    if compressed:
        if os.path.exists(outpath):
            return outpath
    else:
        # Remove the '.gz' extension
        outpath = outpath[:-3] 
        url = url[:-3]
        if os.path.exists(outpath):
            return outpath
        
    # Fetch the file
    try:
        urllib.request.urlretrieve(url, outpath)
        return outpath
    except Exception as e:
        logging.error(f"Failed to fetch {pdbID} from RCSB PDB database.")
        logging.error(e)
        return None
