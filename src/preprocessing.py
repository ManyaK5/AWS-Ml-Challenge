#!/usr/bin/env python3
"""
preprocessing.py — Business record normalisation for entity resolution.

Applies multi-step cleaning to business_name and business_address fields:
- Diacritic stripping (café -> cafe)
- Legal suffix canonicalisation (Private Limited -> pvt ltd)
- Junk prefix removal (***, --, M/s)
- Address abbreviation expansion (St -> street, Ave -> avenue)
- Whitespace/punctuation normalisation
"""

import re
import unicodedata
import pandas as pd


# Pre-compiled regex patterns
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[,;:!?@#$%^&*()\[\]{}<>|/\\=+~`\"]")
_DASH = re.compile(r"[-\u2013\u2014]")
_PREFIX = re.compile(
    r"^\s*(\*+|--|m/s\.?\s*|m/s\s+|##+|the\s+)",
    re.IGNORECASE,
)

# Legal suffix mappings for Indian and US business entities
_LEGAL_MAP = [
    (re.compile(r"\bprivate\s+limited\b", re.I), "pvt ltd"),
    (re.compile(r"\bpvt\.?\s*ltd\.?\b",   re.I), "pvt ltd"),
    (re.compile(r"\bpvt\.?\s+limited\b",  re.I), "pvt ltd"),
    (re.compile(r"\bprivate\s+ltd\.?\b",  re.I), "pvt ltd"),
    (re.compile(r"\blimited\b",           re.I), "ltd"),
    (re.compile(r"\bcorporation\b",       re.I), "corp"),
    (re.compile(r"\bincorporated\b",      re.I), "inc"),
    (re.compile(r"\bllimited\b",          re.I), "ltd"),
    (re.compile(r"\blimited\s+liability\s+company\b", re.I), "llc"),
]

# US street abbreviation mappings
_ADDR_ABBREV = [
    (r"\bst\b",   "street"), (r"\bave?\b", "avenue"),
    (r"\brd\b",   "road"),   (r"\bdr\b",   "drive"),
    (r"\bblvd\b", "boulevard"), (r"\bln\b", "lane"),
    (r"\bct\b",   "court"),  (r"\bpl\b",   "place"),
    (r"\bpkwy\b", "parkway"), (r"\bhwy\b", "highway"),
    (r"\bfwy\b",  "freeway"), (r"\bsq\b",  "square"),
]


def strip_diacritics(text):
    """Remove diacritical marks from unicode text."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def normalize_name(raw):
    """
    Normalise a business name for matching.
    
    Handles legal suffixes, diacritics, junk prefixes, punctuation,
    and common variations (dots, brackets, etc.)
    """
    if pd.isna(raw) or raw is None:
        return ""
    s = strip_diacritics(str(raw)).lower()
    s = _PREFIX.sub("", s)
    s = re.sub(r"\[.*?\]", "", s)
    s = _PUNCT.sub(" ", s)
    s = _DASH.sub(" ", s)
    s = re.sub(r"(?<!\w)\.(?!\w)|(?<=\w)\.(?!\w)", " ", s)
    for pat, rep in _LEGAL_MAP:
        s = pat.sub(rep, s)
    s = re.sub(r"\.(com|net|org|in|co)\b", "", s)
    return _WS.sub(" ", s).strip()


def normalize_address(raw):
    """
    Normalise a business address for matching.
    
    Expands abbreviations, removes unit/apt descriptors,
    strips leading zeros from numeric tokens.
    """
    if pd.isna(raw) or raw is None:
        return ""
    s = str(raw)
    if s.lower() in ("nan", "none", "null", ""):
        return ""
    s = strip_diacritics(s).lower()
    for abbr, full in _ADDR_ABBREV:
        s = re.sub(abbr, full, s)
    s = _PUNCT.sub(" ", s)
    s = _DASH.sub(" ", s)
    s = re.sub(r"\.", " ", s)
    s = re.sub(
        r"\b(unit|apt|apartment|suite|ste|#|no|num|number)\b\s*\w*",
        " ", s
    )
    s = re.sub(r"\b0+(\d+)\b", lambda m: str(int(m.group(0))), s)
    return _WS.sub(" ", s).strip()


def normalize_country(raw):
    """Normalise country field to lowercase stripped string."""
    if pd.isna(raw) or raw is None:
        return ""
    return str(raw).strip().lower()


def preprocess_dataframe(df):
    """
    Apply normalisation pipeline to a source DataFrame.
    
    Input columns: entity_id, business_name, business_address, country
    Added columns: norm_name, norm_address, norm_country
    """
    df = df.copy()
    df["norm_name"] = df["business_name"].apply(normalize_name)
    df["norm_address"] = df["business_address"].apply(normalize_address)
    df["norm_country"] = df["country"].apply(normalize_country)
    return df


def load_source_chunked(path, chunk_size=200_000, nrows=None):
    """
    Load a TSV source file in chunks to control memory usage.
    Each chunk is preprocessed before concatenation.
    """
    chunks = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str,
                              chunksize=chunk_size, nrows=nrows):
        chunks.append(preprocess_dataframe(chunk))
    return pd.concat(chunks, ignore_index=True)
