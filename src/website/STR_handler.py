import re
import requests

# --- ID patterns ---
UNIPROT_RE = re.compile(r'(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$', re.I)
AF_ID_RE   = re.compile(r'^AF-([A-Za-z0-9]{6,10})-F\d+(?:-model_v\d+)?(?:\.(?:pdb|cif))?$', re.I)

# --- endpoints ---
AF_API      = "https://alphafold.ebi.ac.uk/api/prediction/"
AF_FILE_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v{ver}.{ext}"

def process_afid(id):
    RE_AF = re.compile(r'^AF-([A-Za-z0-9]{6,10})-F\d+(?:-model_v\d+)?(?:\.(?:pdb|cif))?$', re.I)
    RE_UNIPROT = re.compile(r'(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$', re.I)
    
    match_af = RE_AF.fullmatch(id)
    match_uni = RE_UNIPROT.fullmatch(id)
    # Convert to UniProt
    if match_af:
        uniprot = match_af.group(1).upper()
    elif match_uni:
        uniprot = id.upper()
    else:
        uniprot = None
    
    if uniprot is None:
        return [f"❌ Not found AlphaFold structure {id}", False]
    
    # 1) API check: non-empty JSON list means "exists"
    try:
        r = requests.get(f"{AF_API}{uniprot}", timeout=10)
        if r.ok:
            try:
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    return [f"✅ Valid Alphafold ID {id}", True]
            except ValueError:
                pass  # not JSON → fall back to file HEAD checks
    except requests.RequestException:
        pass

    # 2) File server HEAD check: try model_v4 → v1
    ext = "cif"
    for ver in (4, 3, 2, 1):
        url = AF_FILE_URL.format(uniprot=uniprot, ver=ver, ext=ext)
        try:
            h = requests.head(url, allow_redirects=True, timeout=10)
            if h.ok:
                return [f"✅ Valid Alphafold ID {id}", True]
        except requests.RequestException:
            continue
    return [f"❌ Not found AlphaFold structure {id}", False]
    
def process_pdbid(id):
    RE_PDB = re.compile(r'^[1-9][A-Za-z0-9]{3}$')  # classic 4-char PDB ID
    if not RE_PDB.fullmatch(id):
        return [f"❌ Invalid format PDB ID", False]
    
    url = f"https://data.rcsb.org/rest/v1/core/entry/{id}"
    try:
        r = requests.get(url, timeout=10)
        return [f"✅ Valid PDB ID {id}", r.status_code == 200]
    except requests.RequestException:
        return [f"❌ Not found PDB ID {id} in RCSB", False]

def process_structure_txt(query: str) -> tuple[str, bool]:
    query = query.strip()

    if len(query) == 4:
        return process_pdbid(query)
    else:
        return process_afid(query)
    
    