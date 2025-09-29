# fasta_parser.py
from __future__ import annotations
from typing import Optional, Callable, Iterable, List, Tuple, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import io
import os
import requests

from ..utils.logger import LOGGER

@dataclass
class FastaRecord:
    """A single FASTA entry with UniProt-style fields."""
    id: str                 # UniProt accession, e.g. A0FGR8
    name: str               # Entry name, e.g. ESYT2_HUMAN
    description: str        # Free text description
    seq: str                # Sequence (uppercased by default)
    def __repr__(self) -> str:
        desc = self.description if len(self.description) <= 40 else self.description[:40] + "…"
        seq_preview = self.seq[:10] + ("…" if len(self.seq) > 10 else "")
        return (f"FastaRecord(id={self.id!r}, name={self.name!r}, "
                f"len={len(self.seq)}, desc={desc!r}, seq={seq_preview!r})")

class FastaParser:
    """
    Minimal FASTA reader/writer & editor (UniProt-aware).

    Header parsing rules:
      - UniProt style:   >sp|<Acc>|<EntryName> <Description...>
                         or >tr|<Acc>|<EntryName> <Description...>
        -> id=<Acc>, name=<EntryName>, description=<Description...>
      - Fallback (generic FASTA):
        >FirstToken [optional rest...]
        -> id=FirstToken, name="", description=<optional rest...>
    """
    def __init__(self, path: Optional[str] = None):
        self.records: List[FastaRecord] = []
        self.path: Optional[str] = None
        if path:
            self.read(path)

    # ------------- I/O -------------
    def read(self, path: str, *, uppercase: bool = True, strip_gaps: bool = False) -> int:
        """Read all sequences from a FASTA file. Returns #records."""
        self.records.clear()
        self.path = path
        with open(path, "r", encoding="utf-8") as fh:
            self._read_handle(fh, uppercase=uppercase, strip_gaps=strip_gaps)
        return len(self.records)

    def read_from_string(self, s: str, *, uppercase: bool = True, strip_gaps: bool = False) -> int:
        """Load records from a FASTA string. Returns #records."""
        self.records.clear()
        self.path = None
        self._read_handle(io.StringIO(s), uppercase=uppercase, strip_gaps=strip_gaps)
        return len(self.records)

    def _read_handle(self, handle, *, uppercase: bool, strip_gaps: bool) -> None:
        header: Optional[str] = None
        seq_chunks: List[str] = []
        for line in handle:
            line = line.rstrip("\r\n")
            if not line or line.isspace():
                continue
            if line.startswith(";"):  # classic FASTA comments
                continue
            if line.startswith(">"):
                if header is not None:
                    self._flush_record(header, seq_chunks)
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq = line.strip().replace(" ", "")
                if uppercase:
                    seq = seq.upper()
                if strip_gaps:
                    seq = seq.replace("-", "")
                seq_chunks.append(seq)
        if header is not None:
            self._flush_record(header, seq_chunks)

    @staticmethod
    def _parse_header(header: str) -> Tuple[str, str, str]:
        """
        Parse header into (id, name, description) with UniProt awareness.
        """
        # UniProt style: db|Acc|EntryName [desc...]
        # Accept any 3-part pipe header; check first token looks like 2 letters (sp/tr) or anything.
        if header.count("|") >= 2:
            first, acc, tail = header.split("|", 2)
            # tail is "EntryName rest..." OR "EntryName" only
            if " " in tail:
                entry, desc = tail.split(" ", 1)
            else:
                entry, desc = tail, ""
            # Some headers may include leading db like "sp" or "tr" in `first`;
            # we don't store it (user asked for 4 fields: id, name, description, seq)
            return acc, entry, desc

        # Fallback: first token is id; rest is description
        parts = header.split(None, 1)
        _id = parts[0]
        _desc = parts[1] if len(parts) > 1 else ""
        return _id, "", _desc

    def _flush_record(self, header: str, seq_chunks: List[str]) -> None:
        _id, _name, _desc = self._parse_header(header)
        seq = "".join(seq_chunks)
        if not _id:
            raise ValueError("FASTA header missing identifier.")
        self.records.append(FastaRecord(id=_id, name=_name, description=_desc, seq=seq))

    def save(self, path: str, *, wrap: int = 60, header_style: str = "uniprot") -> None:
        """
        Save current records to a FASTA file.
        Args:
            path: output filepath
            wrap: wrap sequence lines to this width (0/None = no wrap)
            header_style:
                - "uniprot": >sp|{id}|{name} {description}   (uses 'sp' tag; if name is empty, omits it)
                - "simple" : >{id} {name} {description}      (compacts spaces)
        """
        def build_header(r: FastaRecord) -> str:
            name_part = r.name.strip()
            desc_part = r.description.strip()
            if header_style == "uniprot":
                if name_part and desc_part:
                    return f"sp|{r.id}|{name_part} {desc_part}"
                elif name_part:
                    return f"sp|{r.id}|{name_part}"
                elif desc_part:
                    # no name, keep desc after id
                    return f"sp|{r.id}| {desc_part}".rstrip()
                else:
                    return f"sp|{r.id}|"
            elif header_style == "simple":
                pieces = [r.id]
                if name_part:
                    pieces.append(name_part)
                if desc_part:
                    pieces.append(desc_part)
                return " ".join(pieces)
            else:
                raise ValueError("header_style must be 'uniprot' or 'simple'.")

        with open(path, "w", encoding="utf-8") as out:
            for r in self.records:
                header = build_header(r)
                out.write(f">{header}\n")
                if wrap and wrap > 0:
                    for i in range(0, len(r.seq), wrap):
                        out.write(r.seq[i:i+wrap] + "\n")
                else:
                    out.write(r.seq + "\n")

    # ------------- CRUD -------------
    def add(self, seq_id: str, seq: str, *, name: str = "", description: str = "",
            uppercase: bool = True, strip_gaps: bool = False) -> None:
        """Append a new record."""
        if uppercase:
            seq = seq.upper()
        if strip_gaps:
            seq = seq.replace("-", "")
        self.records.append(FastaRecord(seq_id, name, description, seq))

    def get(self, seq_id: str) -> List[FastaRecord]:
        """Return all records whose id matches `seq_id`."""
        return [r for r in self.records if r.id == seq_id]

    def delete(self, seq_id: str, *, first_only: bool = False) -> int:
        """Delete records by id. Returns number removed."""
        removed, new = 0, []
        for r in self.records:
            if r.id == seq_id and (not first_only or removed == 0):
                removed += 1
            else:
                new.append(r)
        self.records = new
        return removed

    def delete_if(self, predicate: Callable[[FastaRecord], bool]) -> int:
        """Delete records matching a predicate; returns number removed."""
        kept, removed = [], 0
        for r in self.records:
            if predicate(r):
                removed += 1
            else:
                kept.append(r)
        self.records = kept
        return removed

    def rename_ids(self, mapping: Dict[str, str]) -> int:
        """Rename IDs (accessions) per mapping; returns number changed."""
        changed = 0
        for r in self.records:
            if r.id in mapping:
                r.id = mapping[r.id]
                changed += 1
        return changed

    # ------------- Utilities -------------
    def remove_duplicates(
        self,
        *,
        by: str = "id+seq",
        keep: str = "first",
        report: bool = False,
    ):
        """
        Remove duplicate records.

        Args:
            by:     one of {'id','seq','id+seq','id+name'}
            keep:   'first' (keep earliest) or 'last' (keep latest)
            report: if True, also return a detailed report of duplicates

        Returns:
            If report=False: int -> number of records removed
            If report=True:  (int, list[dict]) ->
                removed_count, [
                    {
                    'key': tuple,                   # the dedup key (e.g., (id,) or (id,seq))
                    'kept_index': int,              # index in original list
                    'kept_record': FastaRecord,
                    'removed_indices': list[int],   # indices in original list
                    'removed_records': list[FastaRecord],
                    },
                    ...
                ]
        """
        if by not in {"id", "seq", "id+seq", "id+name"}:
            raise ValueError("by must be one of {'id','seq','id+seq','id+name'}")
        if keep not in {"first", "last"}:
            raise ValueError("keep must be 'first' or 'last'")

        def key_fn(r):
            if by == "id":       return (r.id,)
            if by == "seq":      return (r.seq,)
            if by == "id+name":  return (r.id, r.name)
            return (r.id, r.seq)  # id+seq

        # Map dedup key -> list of original indices
        groups = {}
        for idx, rec in enumerate(self.records):
            k = key_fn(rec)
            groups.setdefault(k, []).append(idx)

        kept_idx = set()
        removed_count = 0
        report_list = []

        for k, idxs in groups.items():
            if keep == "first":
                keep_i = idxs[0]
            else:  # keep == "last"
                keep_i = idxs[-1]
            kept_idx.add(keep_i)

            if len(idxs) > 1:
                rem = [i for i in idxs if i != keep_i]
                removed_count += len(rem)
                if report:
                    report_list.append({
                        "key": k,
                        "kept_index": keep_i,
                        "kept_record": self.records[keep_i],
                        "removed_indices": rem,
                        "removed_records": [self.records[i] for i in rem],
                    })

        # Rebuild in original order using original list
        original = self.records
        self.records = [original[i] for i in range(len(original)) if i in kept_idx]

        if report:
            return removed_count, report_list
        return removed_count

    def subset_by_ids(self, ids: Iterable[str]) -> "FastaParser":
        """Return a new FastaParser with only records whose id is in `ids`."""
        ids = set(ids)
        fp = FastaParser()
        fp.records = [FastaRecord(r.id, r.name, r.description, r.seq) for r in self.records if r.id in ids]
        return fp

    def filter(self, *, min_len: Optional[int] = None, max_len: Optional[int] = None,
               id_contains: Optional[str] = None, name_contains: Optional[str] = None,
               desc_contains: Optional[str] = None) -> "FastaParser":
        """Return a new FastaParser filtered by simple predicates."""
        def ok(r: FastaRecord) -> bool:
            if min_len is not None and len(r.seq) < min_len: return False
            if max_len is not None and len(r.seq) > max_len: return False
            if id_contains and id_contains not in r.id: return False
            if name_contains and name_contains not in r.name: return False
            if desc_contains and desc_contains not in r.description: return False
            return True
        fp = FastaParser()
        fp.records = [FastaRecord(r.id, r.name, r.description, r.seq) for r in self.records if ok(r)]
        return fp

    # Python niceties
    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, i: int) -> FastaRecord:
        return self.records[i]

    def __repr__(self) -> str:
        preview_ids = [r.id for r in self.records[:5]]
        more = len(self.records) - len(preview_ids)
        path_part = f", path={self.path!r}" if getattr(self, "path", None) else ""
        more_part = f", … +{more} more" if more > 0 else ""
        return f"FastaParser(n={len(self.records)}, ids={preview_ids!r}{more_part}{path_part})"




def _fetch_one(
    acc: str,
    session: requests.Session,
    *,
    base_url: str = "https://rest.uniprot.org/uniprotkb/{acc}.fasta",
    timeout: float = 15.0,
) -> Tuple[str, bool, str, int, str]:
    """
    Fetch a single UniProt FASTA.

    Returns:
        (acc, ok, text, http_status, message)
        - ok=True when a FASTA is returned (starts with '>').
        - text='' on failure.
    """
    url = base_url.format(acc=acc)
    try:
        r = session.get(url, timeout=timeout, headers={"Accept": "text/plain"})
        if r.ok and r.text.lstrip().startswith(">"):
            return acc, True, r.text, r.status_code, "OK"
        return acc, False, "", r.status_code, f"HTTP {r.status_code}"
    except requests.RequestException as e:
        return acc, False, "", 0, str(e)


def fetch_fasta(
    accs: Union[str, Iterable[str]],
    *,
    folder: str = ".",
    filename: str = "output",
    refresh: bool = False,
    threads: int = 8,
    timeout: float = 15.0,
    base_url: str = "https://rest.uniprot.org/uniprotkb/{acc}.fasta",
) -> Tuple[str, Dict[str, Any]]:
    """
    Fetch one or multiple UniProt sequences and save into a single FASTA.

    Args:
        accs:      accession or list/iterable of accessions
        folder:    output directory
        filename:  output file name (without extension)
        refresh:   overwrite existing <filename>.fasta if True
        threads:   number of concurrent workers
        timeout:   per-request timeout (seconds)
        base_url:  URL template; must contain '{acc}'

    Returns:
        (output_path, summary)
        summary = {
          "ok": [acc, ...],                # in write order
          "failed": [{"acc","status","message"}, ...],
          "n_ok": int,
          "n_failed": int
        }
    """
    # normalize input and preserve order while deduplicating
    if isinstance(accs, str):
        acc_list: List[str] = [accs]
    else:
        seen, acc_list = set(), []
        for a in accs:
            if a not in seen:
                seen.add(a)
                acc_list.append(a)

    os.makedirs(folder, exist_ok=True)
    out_fa = os.path.join(folder, f"{filename}.fasta")
    manifest = os.path.join(folder, f"{filename}.manifest.tsv")

    if os.path.exists(out_fa) and not refresh:
        LOGGER.info(f"{out_fa} already exists (use refresh=True to overwrite).")
        return out_fa, {"ok": [], "failed": [], "n_ok": 0, "n_failed": 0}

    if not acc_list:
        with open(out_fa, "w", encoding="utf-8"):
            pass
        LOGGER.info(f"Nothing to fetch. Created empty file: {out_fa}")
        return out_fa, {"ok": [], "failed": [], "n_ok": 0, "n_failed": 0}

    LOGGER.info(f"Fetching {len(acc_list)} accession(s) with {threads} threads...")

    ok_texts: Dict[str, str] = {}
    failed: List[Dict[str, Any]] = []

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=max(1, int(threads))) as ex:
            futs = {
                ex.submit(_fetch_one, acc, session, base_url=base_url, timeout=timeout): acc
                for acc in acc_list
            }
            for fut in as_completed(futs):
                acc = futs[fut]
                a, ok, text, status, msg = fut.result()
                if ok:
                    ok_texts[acc] = text
                    LOGGER.info(f"Fetched: {acc}")
                else:
                    failed.append({"acc": acc, "status": status, "message": msg})
                    LOGGER.info(f"Failed:  {acc} ({msg})")

    # write successes in original requested order
    ok_ordered = [a for a in acc_list if a in ok_texts]
    with open(out_fa, "w", encoding="utf-8") as f:
        for acc in ok_ordered:
            f.write(ok_texts[acc].rstrip() + "\n")

    # simple manifest
    with open(manifest, "w", encoding="utf-8") as mf:
        mf.write("accession\tstatus\tmessage\n")
        for acc in ok_ordered:
            mf.write(f"{acc}\tOK\t-\n")
        for f in failed:
            mf.write(f"{f['acc']}\tFAIL\t{f['message']}\n")

    LOGGER.info(f"Saved {len(ok_ordered)} sequences to: {out_fa}")
    if failed:
        LOGGER.info(f"{len(failed)} accession(s) failed. Manifest: {manifest}")

    summary = {"ok": ok_ordered, "failed": failed, "n_ok": len(ok_ordered), "n_failed": len(failed)}
    return out_fa, summary