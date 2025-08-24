import os, re, unicodedata
import gradio as gr

# 20 canonical amino acids
AA = "ACDEFGHIKLMNPQRSTVWY"

# Accept compact or spaced SAV formats
RE_COMPACT = re.compile(rf"""
    ^\s*
    (?P<id>[A-Z0-9]{{6,10}})      # UniProt-like id
    \s+
    (?P<wt>[{AA}])                   # wild-type
    (?P<pos>[1-9]\d*)                # positive integer
    (?P<mut>[{AA}])                  # mutant
    \s*$
""", re.I | re.X)

RE_SPACED = re.compile(rf"""
    ^\s*
    (?P<id>[A-Z0-9]{{6,10}})
    \s+
    (?P<pos>[1-9]\d*)
    \s+
    (?P<wt>[{AA}])
    \s+
    (?P<mut>[{AA}])
    \s*$
""", re.I | re.X)

# Split entries by comma / semicolon / period / newline / tabs / pipes
ENTRY_SPLIT = re.compile(r"[,\.;\|\t\r\n]+")

def _read_text_file_safely(path: str):
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1", errors="replace")

def process_sav_txt(query: str):
    # re.I to allow lowercase AA, eg. O14508 S52N
    SAV_format = re.compile(r'^([A-Za-z0-9]{6,10})\s+([ACDEFGHIKLMNPQRSTVWY])([1-9]\d*)([ACDEFGHIKLMNPQRSTVWY])$', re.I)
    UniProt_format = re.compile(r'^([A-Za-z0-9]{6,10})$', re.I)
    
    parts = re.split(r'[,\n;\t]+', query)
    parts = [e.strip() for e in parts if e.strip()]

    # 1 SAV / 1 UniProtID / 1 (UniProtID + resID)
    if len(parts) == 1:
        tokens = parts[0].split()
        # [A-Za-z0-9]{6,10}
        if len(tokens) == 1 and UniProt_format.fullmatch(tokens[0]):
            return [parts, f"✅ Received 1 query: {parts[0]}", True]
        elif len(tokens) == 2 and tokens[1].isnumeric():
            # Second token is residue id. eg O14508 52
            return [parts, f"✅ Received 1 query: {parts[0]}", True]
        elif len(tokens) == 2 and SAV_format.fullmatch(parts[0]):
            # Second token is mutation point. eg. O14508 S52N
            return [parts, f"✅ Received 1 query: {parts[0]}", True]
        else:
            return [parts, f"❌ Invalid format for: {parts[0]}", False]
    
    # Multiple SAVs
    if len(parts) > 100:
        gr.Warning(f"You entered {len(parts)} SAVs; maximum is 100.")
        return [parts, "⚠️ Too many entries (max 100 allowed).", False]
    
    # Validate each entry format
    invalids = [p for p in parts if not SAV_format.fullmatch(p)]
    if invalids:
        invalids = ''.join(invalids)
        return [
            parts,
            f"❌ Invalid format for: {invalids:.20s}....\n\nEach SAV must be \<UniProt ID\> \<mutation site\>", 
            False
        ]

    return [parts, f"✅ Received {len(parts)} valid SAV entries.", True]

def parse_one(entry):
    """Return (ok, (id, pos, wt, mut)) for a single token."""
    try:
        m = RE_COMPACT.match(entry) or RE_SPACED.match(entry)
        g = m.groupdict()
        return g["id"], int(g["pos"]), g["wt"], g["mut"]
    except:
        return None

def handle_sav_input(SAV_input):
    if SAV_input and os.path.isfile(SAV_input):
        text = _read_text_file_safely(SAV_input)
    else:
        text = str(SAV_input or "")

    # Normalize, strip comments (# ...), trim whitespace
    text = unicodedata.normalize("NFKC", text)
    text = text.upper()
    text = re.sub(r"#.*", "", text)           # remove inline comments
    text = text.strip()

    out = process_sav_txt(text)[0]
    if len(out) == 1:
        parsed = parse_one(out[0])
        if parsed: # e.g. ['O14508 D434A'] 
            uid, pos, wt, mut = parsed
            return [f"{uid} {pos} {wt} {mut}"]
        else: # e.g. ['O14508 434'] / ['O14508']
            return out[0]
    else: # e.g ['O14508 D434A', 'O14508 D434E'] 
        savs_norm = []
        for token in out:
            parsed = parse_one(token)
            uid, pos, wt, mut = parsed
            savs_norm.append(f"{uid} {pos} {wt} {mut}")
        return savs_norm